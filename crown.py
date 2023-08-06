import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten
from adam_optimizer import AdamClipping


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
                # define independent alpha_by_layer for each backward
                self.alpha_by_layer[linear_id] = [None] * (linear_id + 1)

                # define independent alpha for each backward
                alpha_length = sum([linear_layer.out_features for linear_layer in self.seq_model[:linear_id]
                                    if isinstance(linear_layer, nn.Linear)])
                self.alpha[linear_id] = torch.zeros(alpha_length)

        for layer_id in range(len(self.seq_model)):
            if isinstance(self.seq_model[layer_id], nn.Linear):
                pt = 0
                for backward_id in range(layer_id + 1):
                    if isinstance(self.seq_model[backward_id], nn.ReLU):
                        layer_width = self.seq_model[backward_id - 1].out_features
                        self.alpha[layer_id][pt: pt + layer_width] = self.initial_alpha_by_layer[backward_id].clone()
                        pt += layer_width
        for alpha in self.alpha:
            if not alpha is None:
                alpha.requires_grad_()

        for layer_id in range(len(self.seq_model)):
            if isinstance(self.seq_model[layer_id], nn.Linear):
                pt = 0
                for backward_id in range(layer_id + 1):
                    if isinstance(self.seq_model[backward_id], nn.ReLU):
                        layer_width = self.seq_model[backward_id - 1].out_features
                        self.alpha_by_layer[layer_id][backward_id] = self.alpha[layer_id][pt: pt + layer_width]
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
                    D_low = self.alpha_by_layer[layer_id][backward_id]
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
                for layer_id in range(4, self.model_depth):
                    if not self.alpha[layer_id] is None:
                        opt[layer_id] = AdamClipping(params=[self.alpha[layer_id]], lr=lr)
            else:
                for layer_id in range(4, 5):
                    lbs, ubs = self.lbs[layer_id], self.ubs[layer_id]
                    if lbs is None or ubs is None:
                        continue
                    loss = torch.sum(lbs[-1] - ubs[-1])  # the greater the better
                    loss.backward()
                    opt[layer_id].step(clipping=True, lower_limit=torch.zeros_like(self.alpha[layer_id]),
                            upper_limit=torch.ones_like(self.alpha[layer_id]), sign=1)
                    opt[layer_id].zero_grad(set_to_none=True)
                    print(f"iter: {iter}, layer_id: {layer_id}, loss: {loss}")
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
    model.load_state_dict(torch.load('model.pth'))

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    torch.manual_seed(14)
    x = torch.rand(input_width)
    print("output: {}".format(model(x)))
    eps = 1

    boundedmodel = CROWN(model, x, eps)

    print("%%%%%%%%%%%%%%%%%%%%%%% My IBP %%%%%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.IBP()
    for j in range(output_width):
        print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
            j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    print()

    print("%%%%%%%%%%%%%%%%%%%% My CROWN-IBP %%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.crown_IBP()
    for j in range(output_width):
        print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
            j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    print()

    print("%%%%%%%%%%%%%%%%%%%%%%% My CROWN %%%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.crown()
    for j in range(output_width):
        print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
            j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    print()

    print("%%%%%%%%%%%%%%%%%%% My alpha-CROWN %%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.alpha_crown(iteration=200, lr=1e-2)
    for j in range(output_width):
        print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
            j=j, l=lbs[-1][j].item(), u=ubs[-1][j].item()))
    print()

    # print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    # image = x.unsqueeze(0)
    # lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device,
    #                             bound_opts={'sparse_intermediate_bounds': False,
    #                                         'sparse_features_alpha': False})
    # norm = float("inf")
    # ptb = PerturbationLpNorm(norm=norm, eps=eps)
    # image = BoundedTensor(image, ptb)

    # for method in [
    #     'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)', 'CROWN-Optimized']:
    #     print('Bounding method:', method)
    #     if 'Optimized' in method:
    #         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can
    #         # increase verbosity to see per-iteration loss values.
    #         lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
    #     lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
    #     for i in range(1):
    #         for j in range(output_width):
    #             print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
    #                 j=j, l=lb[i][j].item(), u=ub[i][j].item()))
    #     print()
