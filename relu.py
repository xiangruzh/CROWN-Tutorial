import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BoundReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(BoundReLU, self).__init__(inplace)

    @staticmethod
    def convert(act_layer):
        r"""Convert a ReLU layer to BoundReLU layer

        Args:
            act_layer (nn.ReLU): The ReLU layer object to be converted.

        Returns:
            l (BoundReLU): The converted layer object.
        """
        l = BoundReLU(act_layer.inplace)
        return l
    
    def forward(self, x):
        r"""Overwrite the forward function to set the shape of the node
            during a forward pass
        """
        self.shape = x.shape
        return F.relu(x)
    
    def bound_backward(self, last_uA, last_lA, start_node=None, optimize=False):
        r"""Backward propagate through the ReLU layer.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is backward-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is backward-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this backward propagation. It's used for selecting alphas.

            optimize (bool): Indicating whether we are optimizing parameters (alpha).

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.
            
            ubias (tensor): The bias (for upper bound) produced by this layer.
            
            lA( tensor): The new A for computing the lower bound after taking this layer into account.
            
            lbias (tensor): The bias (for lower bound) produced by this layer.
        """
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
            # selected_alpha has shape (2, dim_of_start_node, batch_size=1, dim_of_this_node)
            selected_alpha = self.alpha[start_node]
            if last_lA is not None:
                lb_lower_d = selected_alpha[0].permute(1, 0, 2)
            if last_uA is not None:
                ub_lower_d = selected_alpha[1].permute(1, 0, 2)
        else:
            lb_lower_d = ub_lower_d = (upper_d > 0.5).float()   # CROWN lower bounds
            # Save lower_d as initial alpha for optimization
            self.init_d = lb_lower_d.squeeze(1) # No need to save the extra dimension.
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
        r"""Initialize self.alpha with lower_d that are already saved at
        self.init_d during the initial CROWN backward propagation.

        Args:
            start_nodes (list): A list of start_node, each start_node is a dictionary
            {'idx', 'node'}. 'idx' is an integer indicating the position of the start node,
            while 'node' is the object of the start node.
        """
        self.alpha = OrderedDict()
        alpha_shape = self.shape
        alpha_init = self.init_d
        for start_node in start_nodes:
            ns = start_node['idx']
            size_s = start_node['node'].out_features
            self.alpha[ns] = torch.empty([2, size_s, *alpha_shape]).to(alpha_init) # The first diminsion of alpha_shape is batch size.
            # Why 2? One for the upper bound and one for the lower bound.
            self.alpha[ns].data.copy_(alpha_init.data)
    
    def clip_alpha(self):
        r"""Clip alphas after an single update.
        Alpha should be bewteen 0 and 1.
        """
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0, 1)