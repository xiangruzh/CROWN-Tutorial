# CROWN
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