import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Model
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten


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

        self.lbs = [None] * len(self.seq_model)
        self.ubs = [None] * len(self.seq_model)

    def sequential_backward(self, layer_id, neuron_id, sign=1):
        # For computing the bound of x_{layer_id, neuron_id}
        # return the lower-bounded linear approximation a_all * x + b_all
        # sign=1 by default computes the lower bound, sign=-1 is for (the negative value of) the upper bound

        l = self.seq_model[layer_id]
        a_all = l.weight.data[neuron_id, :] * sign
        b_all = l.bias.data[neuron_id] * sign

        # start backward
        for backward_id in range(layer_id - 1, -1, -1):
            backward_l = self.seq_model[backward_id]
            if isinstance(backward_l, nn.ReLU):
                # backward for ReLU layer
                pre_lb = self.lbs[backward_id - 1]
                pre_ub = self.ubs[backward_id - 1]
                layer_width = a_all.shape[0]
                D = torch.ones(layer_width)
                b = torch.zeros(layer_width)
                for i in range(layer_width):
                    if pre_ub[i] <= 0:
                        D[i] = 0
                    elif pre_lb[i] < 0:
                        # D[i, 0] = pre_ub[i] / (pre_ub[i] - pre_lb[i])  # use parallel slope
                        if a_all[i] >= 0:
                            D[i] = 0 if pre_lb[i] + pre_ub[i] < 0 else 1  # use adaptive slope
                        else:
                            D[i] = pre_ub[i] / (pre_ub[i] - pre_lb[i])
                        b[i] = 0 if a_all[i] >= 0 else -pre_ub[i] * pre_lb[i] / (pre_ub[i] - pre_lb[i])
                b_all = a_all.dot(b) + b_all
                a_all = a_all * D
            elif isinstance(backward_l, nn.Linear):
                # backward for Linear layer
                b_all = a_all.matmul(backward_l.bias.data) + b_all
                a_all = a_all.matmul(backward_l.weight.data)
            else:
                raise RuntimeError("Unsupported network structure")

        return a_all, b_all

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
                A_lb = torch.zeros(l.out_features, self.input_width)
                b_lb = torch.zeros(l.out_features)
                for neuron_id in range(l.out_features):
                    A_lb[neuron_id, :], b_lb[neuron_id] = self.sequential_backward(layer_id, neuron_id, sign=1)
                lb, _ = boundlinear(self.x_lb, self.x_ub, A_lb, b_lb)

                # upper bound
                A_ub = torch.zeros(l.out_features, self.input_width)
                b_ub = torch.zeros(l.out_features)
                for neuron_id in range(l.out_features):
                    A_ub[neuron_id, :], b_ub[neuron_id] = self.sequential_backward(layer_id, neuron_id, sign=-1)
                neg_ub, _ = boundlinear(self.x_lb, self.x_ub, A_ub, b_ub)

                self.lbs[layer_id] = lb
                self.ubs[layer_id] = -neg_ub

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

        # backward (the last layer)
        layer_id += 1
        l = self.seq_model[layer_id]
        # lower bound
        A_lb = torch.zeros(l.out_features, self.input_width)
        b_lb = torch.zeros(l.out_features)
        for neuron_id in range(l.out_features):
            A_lb[neuron_id, :], b_lb[neuron_id] = self.sequential_backward(layer_id, neuron_id, sign=1)
        lb, _ = boundlinear(self.x_lb, self.x_ub, A_lb, b_lb)

        # upper bound
        A_ub = torch.zeros(l.out_features, self.input_width)
        b_ub = torch.zeros(l.out_features)
        for neuron_id in range(l.out_features):
            A_ub[neuron_id, :], b_ub[neuron_id] = self.sequential_backward(layer_id, neuron_id, sign=-1)
        neg_ub, _ = boundlinear(self.x_lb, self.x_ub, A_ub, b_ub)

        self.lbs[layer_id] = lb
        self.ubs[layer_id] = -neg_ub

        return self.lbs, self.ubs


if __name__ == '__main__':
    model = Model()
    # torch.save(model.state_dict(), 'model.pth')
    # model.load_state_dict(torch.load('model.pth'))
    input_width = model.model[0].in_features

    x = torch.rand(input_width)
    print("output: {}".format(model(x)))
    eps = 1

    boundedmodel = CROWN(model, x, eps)

    print("%%%%%%%%%%%%%%%%%%%%%%% My IBP %%%%%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.IBP()

    print("lower bound: {}".format(lbs[-1]))
    print("upper bound: {}".format(ubs[-1]))
    print()

    print("%%%%%%%%%%%%%%%%%%%%%%% My CROWN-IBP %%%%%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.crown_IBP()

    print("lower bound: {}".format(lbs[-1]))
    print("upper bound: {}".format(ubs[-1]))
    print()

    print("%%%%%%%%%%%%%%%%%%%%%%% My CROWN %%%%%%%%%%%%%%%%%%%%%%%%")
    lbs, ubs = boundedmodel.crown()

    print("lower bound: {}".format(lbs[-1]))
    print("upper bound: {}".format(ubs[-1]))
    print()

    print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    image = x.unsqueeze(0)
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
    norm = float("inf")
    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image = BoundedTensor(image, ptb)

    for method in [
        'IBP', 'IBP+backward (CROWN-IBP)', 'backward (CROWN)']:
        print('Bounding method:', method)
        if 'Optimized' in method:
            # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can
            # increase verbosity to see per-iteration loss values.
            lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1}})
        lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0])
        for i in range(1):
            for j in range(2):
                print('f_{j}(x_0): {l:8.4f} <= f_{j}(x_0+delta) <= {u:8.4f}'.format(
                    j=j, l=lb[i][j].item(), u=ub[i][j].item()))
        print()
