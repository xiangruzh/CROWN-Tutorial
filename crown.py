import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model import Model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from adam_optimizer import AdamClipping
from collections import OrderedDict
from contextlib import ExitStack


class BoundLinear(nn.Linear):
    def __init(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l
    
    def bound_backward(self, last_uA, last_lA, start_node=None, optimize=False):
        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            # propagate A to the nest layer
            next_A = last_A.matmul(self.weight)
            # compute the bias of this layer
            sum_bias = last_A.matmul(self.bias)
            return next_A, sum_bias
        uA, ubias = _bound_oneside(last_uA)
        lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias
    
    def interval_propagate(self, h_U, h_L):
        weight = self.weight
        bias = self.bias
        # Linf norm
        mid = (h_U + h_L) / 2.0
        diff = (h_U - h_L) / 2.0
        weight_abs = weight.abs()
        center = torch.addmm(bias, mid, weight.t())
        deviation = diff.matmul(weight_abs.t())
        upper = center + deviation
        lower = center - deviation
        return upper, lower
    

class BoundReLU(nn.ReLU):
    def __init__(self, prev_layer, inplace=False):
        super(BoundReLU, self).__init__(inplace)

    # Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    @staticmethod
    def convert(act_layer, prev_layer):
        l = BoundReLU(prev_layer, act_layer.inplace)
        return l
    
    # Overwrite the forward function to set the shape of the node
    # during a forward pass
    def forward(self, x):
        self.shape = x.shape
        return F.relu(x)
    
    def bound_backward(self, last_uA, last_lA, start_node=None, optimize=False):
        # lb_r and ub_r are the bounds of input (pre-activation)
        lb_r = self.lower_l.clamp(max=0)
        ub_r = self.upper_u.clamp(min=0)
        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)
        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        upper_d = upper_d.unsqueeze(1)
        if optimize:
            selected_alpha = self.alpha[start_node]
            if last_lA is not None:
                lb_lower_d = selected_alpha[0].permute(1, 0, 2)
            if last_uA is not None:
                ub_lower_d = selected_alpha[1].permute(1, 0, 2)
        else:
            lb_lower_d = ub_lower_d = (upper_d > 0.5).float()
            # Save lower_d as initial alpha for optimization
            self.init_d = lb_lower_d
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            uA = upper_d * pos_uA + ub_lower_d * neg_uA
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lb_lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
        return uA, ubias, lA, lbias
    
    def interval_propagate(self, h_U, h_L):
        # stored upper and lower bounds
        self.upper_u = h_U
        self.lower_l = h_L
        return F.relu(h_U), F.relu(h_L)
    
    def init_opt_parameters(self, start_nodes):
        self.alpha = OrderedDict()
        alpha_shape = self.shape
        alpha_init = self.init_d
        for start_node in start_nodes:
            ns = start_node['idx']
            size_s = start_node['node'].out_features
            self.alpha[ns] = torch.empty([2, size_s, 1, *alpha_shape])
            self.alpha[ns].data.copy_(alpha_init.data)
    
    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0, 1)


class BoundSequential(nn.Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)
    
    # Convert a Pytorch model to a model with bounds
    # @param seq_model Input pytorch model
    # @return Converted model
    @staticmethod
    def convert(seq_model):
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l, layers[-1]))
        return BoundSequential(*layers)
    
    # Full CROWN bounds with all intermediate layer bounds computed by CROWN
    def full_backward_range(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                # We set C as the weight of previous layer
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i-1].out_features).unsqueeze(0)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.backward_range(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True, start_node=i-1, optimize=optimize)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.backward_range(x_U=x_U, x_L=x_L, C=torch.eye(modules[i].out_features).unsqueeze(0), upper=upper, lower=lower, start_node=i, optimize=optimize)

    def backward_range(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None, optimize=False):
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
        # Do a forward pass to set perturbed nodes
        self(*x_U)
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
    

def boundlinear(in_lb, in_ub, A, b):
    # compute the bound of the output of a linear transformation

    b = b.unsqueeze(1)
    x_middle = ((in_ub + in_lb) / 2)
    eps = (in_ub - x_middle)

    # view x_middle and eps as matrices for multiplication
    x_middle = x_middle.unsqueeze(1)
    eps = eps.unsqueeze(1)

    y0 = A.matmul(x_middle) + b
    out_eps = torch.abs(A).matmul(eps)
    out_ub = y0 + out_eps
    out_lb = y0 - out_eps

    return out_lb.squeeze(1), out_ub.squeeze(1)


class CROWN:
    def __init__(self, model, x, eps):
        self.model = model
        self.x = x
        self.eps = eps

        self.seq_model = model.model
        self.input_width = self.seq_model[0].in_features

        self.x_lb = x - torch.ones_like(x) * eps
        self.x_ub = x + torch.ones_like(x) * eps

        self.model_depth = len(self.seq_model)

        self.lbs = [None] * self.model_depth
        self.ubs = [None] * self.model_depth

        # define alpha
        self.initial_alpha_by_layer = [None] * self.model_depth


    def initialize_alpha(self):
        # the initial CROWN alpha is stored in initial_alpha_by_layer after the first iteration
        self.alpha_by_layer = [None] * self.model_depth
        self.alpha = [None] * self.model_depth
        for linear_id in range(self.model_depth):
            if isinstance(self.seq_model[linear_id], nn.Linear):
                # define independent alpha for each backward
                alpha_length = sum([linear_layer.out_features for linear_layer in self.seq_model[:linear_id]
                                    if isinstance(linear_layer, nn.Linear)])
                self.alpha[linear_id] = torch.zeros(2, self.seq_model[linear_id].out_features, alpha_length)

        for layer_id in range(len(self.seq_model)):
            if isinstance(self.seq_model[layer_id], nn.Linear):
                layer_width = self.seq_model[layer_id].out_features
                pt = 0
                for backward_id in range(layer_id + 1):
                    if isinstance(self.seq_model[backward_id], nn.ReLU):
                        backward_layer_width = self.seq_model[backward_id - 1].out_features
                        
                        self.alpha[layer_id][:, :, pt: pt + backward_layer_width] = self.initial_alpha_by_layer[backward_id].clone().repeat(2, layer_width, 1)
                        pt += backward_layer_width
        for alpha in self.alpha:
            if not alpha is None:
                alpha.requires_grad_()


        pt = 0
        for backward_id in range(self.model_depth):
            if isinstance(self.seq_model[backward_id], nn.ReLU):
                layer_width = self.seq_model[backward_id - 1].out_features
                self.alpha_by_layer[backward_id] = list(range(pt, pt + layer_width))
                pt += layer_width

    def sequential_backward_layer(self, layer_id, sign=1, optimize_alpha=False):
        # For computing the bound of x_{layer_id}
        # return the lower-bounded linear approximation A_all * x + b_all
        # sign=1 by default computes the lower bound, sign=-1 is for (the negative value of) the upper bound
        # optimize_alpha=True: use alpha stored in self.alpha for optimization;
        #                False: use original CROWN bound

        l = self.seq_model[layer_id]
        A_all = l.weight.data * sign
        b_all = l.bias.data * sign

        # start backward
        for backward_id in range(layer_id - 1, -1, -1):
            backward_l = self.seq_model[backward_id]
            if isinstance(backward_l, nn.ReLU):
                # backward for ReLU layer

                # stable neurons
                pre_lb = self.lbs[backward_id - 1].clamp(max=0)
                pre_ub = self.ubs[backward_id - 1].clamp(min=0)
                pre_ub = torch.max(pre_ub, pre_lb + 1e-8)  # in case pre_ub = pre_lb = 0

                # linear bounds for unstable neurons
                D_up = pre_ub / (pre_ub - pre_lb)
                if optimize_alpha:
                    if sign == 1:
                        D_low = self.alpha[layer_id][0, :, self.alpha_by_layer[backward_id]]
                    else:
                        D_low = self.alpha[layer_id][1, :, self.alpha_by_layer[backward_id]]
                else:
                    D_low = (D_up > 0.5).float()
                    self.initial_alpha_by_layer[backward_id] = D_low

                # We are going to decide the choice of upper and lower bounds according to the signs of a_all
                pos_A_all = torch.clamp(A_all, min=0)
                neg_A_all = torch.clamp(A_all, max=0)

                b_all = neg_A_all.matmul(-pre_lb * D_up) + b_all
                A_all = pos_A_all * D_low + neg_A_all * D_up

            elif isinstance(backward_l, nn.Linear):
                # backward for Linear layer
                b_all = A_all.matmul(backward_l.bias.data) + b_all
                A_all = A_all.matmul(backward_l.weight.data)
            else:
                raise RuntimeError("Unsupported network structure")

        return A_all, b_all

    def crown(self):
        # clear existing bounds
        self.lbs = [None] * len(self.seq_model)
        self.ubs = [None] * len(self.seq_model)

        # The first linear layer
        l = self.seq_model[0]
        A = l.weight.data
        b = l.bias.data
        lb, ub = boundlinear(self.x_lb, self.x_ub, A, b)
        self.lbs[0] = lb
        self.ubs[0] = ub

        for layer_id in range(1, len(self.seq_model)):
            l = self.seq_model[layer_id]
            if isinstance(l, nn.Linear):
                # lower bound
                A_lb, b_lb = self.sequential_backward_layer(layer_id, sign=1)
                lb, _ = boundlinear(self.x_lb, self.x_ub, A_lb, b_lb)

                # upper bound
                A_ub, b_ub = self.sequential_backward_layer(layer_id, sign=-1)
                neg_ub, _ = boundlinear(self.x_lb, self.x_ub, A_ub, b_ub)

                self.lbs[layer_id] = lb
                self.ubs[layer_id] = -neg_ub

        return self.lbs, self.ubs

    def alpha_crown(self, iteration=20, lr=1e-2):
        # clear existing bounds
        self.lbs = [None] * len(self.seq_model)
        self.ubs = [None] * len(self.seq_model)

        # The first linear layer
        l = self.seq_model[0]
        A = l.weight.data
        b = l.bias.data
        lb, ub = boundlinear(self.x_lb, self.x_ub, A, b)
        self.lbs[0] = lb
        self.ubs[0] = ub

        for iter in range(iteration):
            # We use the original CROWN bounds for the first iteration
            # to initialize alpha
            optimize_alpha = False if iter == 0 else True
            for layer_id in range(1, len(self.seq_model)):
                l = self.seq_model[layer_id]
                if isinstance(l, nn.Linear):
                    # lower bound
                    A_lb, b_lb = self.sequential_backward_layer(layer_id, sign=1, optimize_alpha=optimize_alpha)
                    lb, _ = boundlinear(self.x_lb, self.x_ub, A_lb, b_lb)

                    # upper bound
                    A_ub, b_ub = self.sequential_backward_layer(layer_id, sign=-1, optimize_alpha=optimize_alpha)
                    neg_ub, _ = boundlinear(self.x_lb, self.x_ub, A_ub, b_ub)

                    self.lbs[layer_id] = lb
                    self.ubs[layer_id] = -neg_ub
                    
            if iter == 0:
                self.initialize_alpha()
                opt = [None] * self.model_depth
                for layer_id in range(1, self.model_depth):
                    if not self.alpha[layer_id] is None:
                        opt[layer_id] = AdamClipping(params=[self.alpha[layer_id]], lr=lr)
            else:
                for layer_id in range(1, self.model_depth):
                    lbs, ubs = self.lbs[layer_id], self.ubs[layer_id]
                    if lbs is None or ubs is None:
                        continue
                    loss = torch.sum(lbs - ubs)  # the greater the better
                    # loss = torch.sum(lbs - ubs)  # the greater the better
                    # loss = sum(lbs)
                    loss.backward(retain_graph=True)
                    opt[layer_id].step(clipping=True, lower_limit=torch.zeros_like(self.alpha[layer_id]),
                            upper_limit=torch.ones_like(self.alpha[layer_id]), sign=1)
                    opt[layer_id].zero_grad(set_to_none=True)
                    # loss = -sum(ubs)
                    # loss.backward(retain_graph=True)
                    # opt[layer_id].step(clipping=True, lower_limit=torch.zeros_like(self.alpha[layer_id]),
                    #         upper_limit=torch.ones_like(self.alpha[layer_id]), sign=1)
                    # opt[layer_id].zero_grad(set_to_none=True)
                    # print(f"iter: {iter}, layer_id: {layer_id}, loss: {loss}")
                # print()

        return self.lbs, self.ubs

    def IBP(self):
        # clear existing bounds
        self.lbs = [None] * len(self.seq_model)
        self.ubs = [None] * len(self.seq_model)

        lb = self.x_lb
        ub = self.x_ub
        layer_id = 0
        for l in self.seq_model:
            if isinstance(l, nn.Linear):
                lb, ub = boundlinear(lb, ub, l.weight.data, l.bias.data)
            elif isinstance(l, nn.ReLU):
                lb = F.relu(lb)
                ub = F.relu(ub)
            else:
                raise RuntimeError("Unsupported network structure")
            self.lbs[layer_id] = lb
            self.ubs[layer_id] = ub
            layer_id += 1

        return self.lbs, self.ubs

    def crown_IBP(self):
        # clear existing bounds
        self.lbs = [None] * len(self.seq_model)
        self.ubs = [None] * len(self.seq_model)

        # forward
        lb = self.x_lb
        ub = self.x_ub
        for layer_id in range(len(self.seq_model) - 1):
            l = self.seq_model[layer_id]
            if isinstance(l, nn.Linear):
                lb, ub = boundlinear(lb, ub, l.weight.data, l.bias.data)
            elif isinstance(l, nn.ReLU):
                lb = F.relu(lb)
                ub = F.relu(ub)
            else:
                raise RuntimeError("Unsupported network structure")
            self.lbs[layer_id] = lb
            self.ubs[layer_id] = ub

        # backward (only for the last layer)
        layer_id += 1
        # lower bound
        A_lb, b_lb = self.sequential_backward_layer(layer_id, sign=1)
        lb, _ = boundlinear(self.x_lb, self.x_ub, A_lb, b_lb)

        # upper bound
        A_ub, b_ub = self.sequential_backward_layer(layer_id, sign=-1)
        neg_ub, _ = boundlinear(self.x_lb, self.x_ub, A_ub, b_ub)

        self.lbs[layer_id] = lb
        self.ubs[layer_id] = -neg_ub

        return self.lbs, self.ubs


if __name__ == '__main__':
    model = Model()
    # torch.save(model.state_dict(), 'model_verysimple.pth')
    # model.load_state_dict(torch.load('model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    torch.manual_seed(14)
    x = torch.rand(input_width).unsqueeze(0)
    print("output: {}".format(model(x)))
    eps = 1
    x_u = x + eps
    x_l = x - eps

    boundedmodel = BoundSequential.convert(model.model)
    ub, _ = boundedmodel._get_optimized_bounds(x_L=x_l, x_U=x_u, upper=True, lower=False)
    _, lb = boundedmodel._get_optimized_bounds(x_L=x_l, x_U=x_u, upper=False, lower=True)
    for j in range(output_width):
        print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
            j=j, l=lb[0][j].item(), u=ub[0][j].item()))
    print()

    # boundedmodel = CROWN(model, x, eps)

    # print("%%%%%%%%%%%%%%%%%%%%%%% My IBP %%%%%%%%%%%%%%%%%%%%%%%%%%")
    # lbs, ubs = boundedmodel.IBP()
    # for j in range(output_width):
    #     print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
    #         j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    # print()

    # print("%%%%%%%%%%%%%%%%%%%% My CROWN-IBP %%%%%%%%%%%%%%%%%%%%%%%")
    # lbs, ubs = boundedmodel.crown_IBP()
    # for j in range(output_width):
    #     print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
    #         j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    # print()

    # print("%%%%%%%%%%%%%%%%%%%%%%% My CROWN %%%%%%%%%%%%%%%%%%%%%%%%")
    # lbs, ubs = boundedmodel.crown()
    # for j in range(output_width):
    #     print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
    #         j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    # print()

    # print("%%%%%%%%%%%%%%%%%%% My alpha-CROWN %%%%%%%%%%%%%%%%%%%%%%")
    # lbs, ubs = boundedmodel.alpha_crown(iteration=20, lr=1e-1)
    # for j in range(output_width):
    #     print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
    #         j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    # print()

    print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    image = x
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device,
                                bound_opts={'sparse_intermediate_bounds': False,
                                            'sparse_features_alpha': False})
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb)

    for method in [
        'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can
            # increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(1):
            for j in range(output_width):
                print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
                    j=j, l=lb[i][j].item(), u=ub[i][j].item()))
        print()
