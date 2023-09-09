# A Simple Implementation of CROWN and alpha-CROWN
This is a simple implementation of CROWN and alpha-CROWN algorithms. It's designed to be a simplified version of [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN). Here, we only consider fully connected sequential ReLU networks. 

## Prerequisite
To check the correctness of the bounds computed by this implementation, you need to first install [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA). The results obtained by this implementation (labeled as ```CROWN``` and ```alpha-CROWN```) should be the same as the results obtained by the library (labeled as ```auto_LiRPA: backward (CROWN)``` and ```auto_LiRPA: CROWN-Optimized```).

## Contents
#### ```crown.py```
A simple implementation of CROWN and alpha-CROWN to compute bounds of fully connected sequential ReLU networks. It also uses the library ```auto_LiRPA``` to compute the bounds of the same models for comparison.

#### ```model.py```
The definition of a PyTorch model. All the layers in the model is gathered in an ```nn.Sequential``` object.

#### ```model.pth```
A pretrained PyTorch model for debugging. The structure of it is ```(2, 50, 100, 2)```.

## Run
```
python crown.py
```

## CROWN
### The Algorithm
1. Given an input ```x``` and an L_inf perturbation, we have the upper bound and the lower bound of the input, noted as ```x_U``` and ```x_L``` respectively. Our goal is to compute the lower (or upepr) bound for the output of the network. To do so, we construct a linear function to lower (or upper) bound the original non-linear network. (See Lemma 2.2 of [Wang, S., Zhang, H., Xu, K., Lin, X., Jana, S., Hsieh, C. J., & Kolter, J. Z. (2021). Beta-crown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification. Advances in Neural Information Processing Systems, 34, 29909-29921.](https://arxiv.org/pdf/2103.06624.pdf))

2. To compute a bounded relaxation of the original network, we need to bound each non-linear layer first. Here, we use two linear constraints to relax unstable ReLU neurons: a linear upper bound and a linear lower bound. (See [Lemma 2.1](https://arxiv.org/pdf/2103.06624.pdf))

3. With upper bounds and lower bounds of each ReLU layer, we use backward propagation to compute the upper and lower bound for the whole network.

4. Notice that our ReLU relaxation requires the bounds of its input. Thus we need to compute the bounds for intermediate layers, too. Here, we apply backward propagation starting from every linear intermidiate layers to obtain the intermediate bounds.

### Implementation
1. We wrap every linear layers and ReLU layers into ```BoundLinear``` and ```BoundReLU``` objects. The backward propagations through a single layer are implemented in these classes as ```bound_backward()```. The backward propagation for linear layers is straightforward. The details of the backward propagation for ReLU layers can be seen in [the proof for Lemma 2.1](https://arxiv.org/pdf/2103.06624.pdf).

2. The sequential ReLU model is converted to a ```BoundSequential``` object, where each layer is converted to a corresponding ```BoundLinear``` or ```BoundReLU``` object.

3. ```backward_range()``` of ```BoundSequential``` is the function to compute the output bounds for each linear layer (which can be either the final layer or a layer that followed by a ReLU layer) with a backward propagation starting from it.

4. ```full_backward_range()``` of ```BoundSequential``` iterates through the whole sequential model layer by layer. For each linear layer, it bounds the output by calling ```backward_range()``` with assigning this layer as the ```start_node```. In the end, it bounds the output of the final layer and thus provides the bounds for the whole model.

## alpha-CROWN
### The Algorithm
When using linear relaxation on each ReLU layer, the slopes of the lower bounds for every single neurons can vary. Apart from preset their values, we can take them as variables ```alpha``` and optimize them based on the tightness of final bounds.

### Implementation
1. ```_get_optimized_bounds()``` of ```BoundSequential``` is for computing the optimized bounds of the model with alpha-CROWN. 

2. The first step of the algorithm is to intialize alpha with ```init_alpha()```. There, a full CROWN is run to obtain intermediate bounds and initial alpha. Alphas are saved as properties of ```BoundReLU``` objects. Since we use independent alphas for computing the bound of each intermediate or final neuron, the length of the dictionary of ```alpha``` in a ```BoundReLU``` object equals to the number of linear layers from which we do backward propagations.

3. ```_set_alpha``` is for gathering ```alpha```s from every ```BoundReLU``` layers and construct a list ```parameters``` to be optimized later.

4. We use the current values of ```alpha``` to compute the upper bound (or the lower bound) with CROWN in each iteration. The loss is the sum of ```ub``` or the negative of the sum of ```lb```. We use Adam optimizer to optimize alpha.



