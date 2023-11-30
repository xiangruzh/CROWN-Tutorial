import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model import Model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from collections import OrderedDict
from contextlib import ExitStack
from linear import BoundLinear
from relu import BoundReLU


class BoundSequential(nn.Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)
    
    # Convert a Pytorch model to a model with bounds
    # @param seq_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(seq_model):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.
        
        Returns:
            The converted BoundSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
        return BoundSequential(*layers)
    
    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        r"""Main function for computing bounds.

        Args:
        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.    
        """
        ub = lb = None
        if optimize:
            # alpha-CROWN
            if upper:
                ub, _ = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=True, lower=False)
            if lower:
                _, lb = self._get_optimized_bounds(x_L=x_L, x_U=x_U, upper=False, lower=True)
        else:
            # CROWN
            ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        return ub, lb
    
    # Full CROWN bounds with all intermediate layer bounds computed by CROWN
    def full_backward_range(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        r"""A full backward propagation. We are going to sequentially compute the 
        intermediate bounds for each linear layer followed by a ReLU layer. For each
        intermediate bound, we call self.backward_range() to do a backward propagation 
        starting from that layer.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.    
        """
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i-1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.backward_range(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True, start_node=i-1, optimize=optimize)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper, lower=lower, start_node=i, optimize=optimize)

    def backward_range(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False):
        r"""The backward propagation starting from a given node. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound. 

            start_node (int): The start node of this propagation. It should be a linear layer.

            optimize (bool): Whether we optimize parameters.

        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        # start propagation from the last layer
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node+1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.bound_backward(upper_A, lower_A, start_node, optimize)
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign = -1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound
        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb
    
    def _get_optimized_bounds(self, x_U=None, x_L=None, upper=False, lower=True):
        r"""The main function of alpha-CROWN.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound. 

        Returns:
            best_ret_u (tensor): Optimized upper bound of the final output.
            best_ret_l (tensor): Optimized lower bound of the final output.
        """
        modules = list(self._modules.values())
        self.init_alpha(x_U=x_U, x_L=x_L)
        alphas, parameters = [], []
        best_alphas = self._set_alpha(parameters, alphas, lr=1e-1)
        opt = optim.Adam(parameters)
        # Create a weight vector to scale learning rate.
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.98)
        iteration = 20
        best_intermediate_bounds = {}
        need_grad = True
        for i in range(iteration):
            if i == iteration - 1:
                # No grad update needed for the last iteration
                need_grad = False
            with torch.no_grad() if not need_grad else ExitStack():
                ub, lb = self.full_backward_range(x_U=x_U, x_L=x_L, upper=upper, lower=lower, optimize=True)
            if i == 0:
                # save results at the first iteration
                best_ret = []
                best_ret_l = _save_ret_first_time(lb, float('-inf'), best_ret)
                best_ret_u = _save_ret_first_time(ub, float('inf'), best_ret)
                for node_id, node in enumerate(modules):
                    if isinstance(node, BoundReLU):
                        new_intermediate = [node.lower_l.detach().clone(),
                                            node.upper_u.detach().clone()]
                        best_intermediate_bounds[node_id] = new_intermediate        
            l = lb
            if lb is not None:
                l = torch.sum(lb)
            u = ub
            if ub is not None:
                u = torch.sum(ub)        
            loss_ = l if lower else -u
            loss = (-1 * loss_).sum()
            with torch.no_grad():
                best_ret_l = torch.max(best_ret_l, lb)
                best_ret_u = torch.min(best_ret_u, ub)
                self._update_optimizable_activations(best_intermediate_bounds, best_alphas)
            opt.zero_grad(set_to_none=True)
            if i != iteration - 1:
                # We do not need to update parameters in the last step since the
                # best result already obtained
                loss.backward()
                opt.step()
            for i, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    node.clip_alpha()
            scheduler.step()
        # Set all variables to their saved best values
        with torch.no_grad():
            for idx, node in enumerate(modules):
                if isinstance(node, BoundReLU):
                    # Assigns a new dictionary
                    node.alpha = best_alphas[idx]
                    best_intermediate = best_intermediate_bounds[idx]
                    node.lower_l.data = best_intermediate[0].data
                    node.upper_u.data = best_intermediate[1].data
        return best_ret_u, best_ret_l

    def init_alpha(self, x_U=None, x_L=None):
        r"""Initialize alphas and intermediate bounds for alpha-CROWN
        Contains a full CROWN method.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

        Returns:
            lb (tensor): Lower CROWN bound.

            ub (tensor): Upper CROWN bound.

            init_intermediate_bounds (dictionary): Intermediate bounds obtained 
            by initial CROWN.
        """
        # Do a forward pass to set perturbed nodes
        self(x_U)
        # Do a CROWN to init all intermediate layer bounds and alpha
        ub, lb = self.full_backward_range(x_U, x_L)
        modules = list(self._modules.values())
        # Also collect the initial intermediate bounds
        init_intermediate_bounds = {}
        for i, module in enumerate(modules):
            if isinstance(module, BoundReLU):
                start_nodes = self.get_alpha_crown_start_nodes(i)
                module.init_opt_parameters(start_nodes)
                init_intermediate_bounds[i-1] = [module.lower_l, module.upper_u]
        return lb, ub, init_intermediate_bounds
    
    def _set_alpha(self, parameters, alphas, lr):
        r"""Collect alphas from all the ReLU layers and gather them
        into "parameters" for optimization. Also construct best_alphas
        to keep tracking the values of alphas.

        Args:
            parameters (list): An empty list, to gather all alphas for optimization.

            alphas (list): An empty list, to gather all values of alphas.

            lr (float): Learning rate, for optimization.

        best_alphas (OrderDict): An OrderDict object to collect the value of alpha.
        """
        modules = list(self._modules.values())
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                alphas.extend(list(node.alpha.values()))
        # Alpha has shape (2, output_shape, batch_dim, node_shape)
        parameters.append({'params': alphas, 'lr': lr, 'batch_dim': 2})
        # best_alpha is a dictionary of dictionary. Each key is the alpha variable
        # for one actiation layer, and each value is a dictionary contains all
        # activation layers after that layer as keys.
        best_alphas = OrderedDict()
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                best_alphas[i] = {}
                for alpha_node in node.alpha:
                    best_alphas[i][alpha_node] = node.alpha[alpha_node].detach().clone()
                    node.alpha[alpha_node].requires_grad_()
        return best_alphas

    # For a given node, return the list of indices of its "start_nodes"
    # A "start_node" of a given node is a node from which a backward propagation uses the given node,
    # so we will store a set of alpha for that "start_node" with the given node.
    def get_alpha_crown_start_nodes(self, node_id):
        modules = list(self._modules.values())
        start_nodes = []
        for i in range(node_id, len(modules)):
            if isinstance(modules[i], BoundLinear):
                start_nodes.append({'idx': i, 'node': modules[i]})
        return start_nodes
    
    # Update bounds and alpha of optimizable activations
    def _update_optimizable_activations(self, best_intermediate_bounds, best_alphas):
        modules = list(self._modules.values())
        for i, node in enumerate(modules):
            if isinstance(node, BoundReLU):
                best_intermediate_bounds[i][0] = torch.max(
                    best_intermediate_bounds[i][0],
                    node.lower_l
                )
                best_intermediate_bounds[i][1] = torch.min(
                    best_intermediate_bounds[i][1],
                    node.upper_u
                )
                for alpha_m in node.alpha:
                    best_alphas[i][alpha_m] = node.alpha[alpha_m]
        
    
# Save results at the first iteration to best_ret.
def _save_ret_first_time(bounds, fill_value, best_ret):
    if bounds is not None:
        best_bounds = torch.full_like(bounds, fill_value=fill_value, dtype=torch.float32)
    else:
        best_bounds = None
    if bounds is not None:
        best_ret.append(bounds.detach().clone())
    else:
        best_ret.append(None)
    return best_bounds


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device: {}".format(device))

    model = Model().to(device)
    # torch.save(model.state_dict(), 'model_verysimple.pth')
    # model.load_state_dict(torch.load('model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    torch.manual_seed(14)
    batch_size = 2
    x = torch.rand(batch_size, input_width).to(device)
    print("output: {}".format(model(x)))
    eps = 1
    x_u = x + eps
    x_l = x - eps

    print("%%%%%%%%%%%%%%%%%%%%%%%% CROWN %%%%%%%%%%%%%%%%%%%%%%%%%%")
    boundedmodel = BoundSequential.convert(model.model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
        print('---------------------------------------------------------')
    print()
    
    print("%%%%%%%%%%%%%%%%%%%%% alpha-CROWN %%%%%%%%%%%%%%%%%%%%%%%")
    boundedmodel = BoundSequential.convert(model.model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True, optimize=True)
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
        print('---------------------------------------------------------')
    print()

    print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    image = x
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device,
                                bound_opts={'sparse_intermediate_bounds': False,
                                            'sparse_features_alpha': False})
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb)

    for method in ['backward (CROWN)', 'CROWN-Optimized']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can
            # increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(batch_size):
            for j in range(output_width):
                print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                    j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))
            print('---------------------------------------------------------')
        print()
