import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = Model()

    input_width = model.model[0].in_features
    output_width = model.model[-1].out_features

    x = torch.rand(input_width)
    print("output: {}".format(model(x)))
    eps = 1

    print("%%%%%%%%%%%%%%%%%%%%% auto-LiRPA %%%%%%%%%%%%%%%%%%%%%%%%")
    image = x.unsqueeze(0)
    lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
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
