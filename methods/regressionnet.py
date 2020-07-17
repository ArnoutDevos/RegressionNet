# This code is authored by [anonymous]

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class RegressionNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, lamb):
        super(RegressionNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_S_norms = None
        self.keep_record = True
        
        # Custom params
        self.lamb = lamb # Training regularization parameter
        print("Initialized RegressionNet with regularization hyperparameter: {}".format(self.lamb))

    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        # Work on support vectors
        z_support   = z_support.contiguous()
        #z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(dim=1) #the shape of z is [n_data, n_dim], dim=1 is dimension to reduce (take mean)
        z_support_view     = z_support.view(self.n_way, self.n_support, -1 ) #the shape of z is [n_classes, n_support, n_dim]

        
        # Batched inverses :), in PyTorch > 0.4.1
        # Need regularization to ensure matrix inverse is possible to compute
        z_supports_inv = torch.matmul(torch.inverse(torch.matmul(z_support_view, z_support_view.transpose(1,2)) + 1e-3*torch.eye(self.n_support).cuda().repeat(self.n_way, 1 , 1)), z_support_view)
        
        # Work on query vectors
        z_query     = z_query.contiguous().view(self.n_way*self.n_query, -1 ) # stack em all, into [n_way*n_query, n_dim
        beta_hat = torch.matmul(z_supports_inv, z_query.transpose(0,1)) # ? [n_classes, n_support, n_way*n_query] ([n_classes, n_support, n_dim] * [n_dim, n_way*n_query])
        z_lrc = torch.matmul(z_support_view.transpose(1,2), beta_hat)# ? [n_classes, n_way*n_query, n_support] ([n_classes, n_dim, n_support] * [n_classes, n_support, n_way*n_query])
        
        # z_query = [n_way*n_query, n_dim]
        # z_lrc = [n_classes, n_dim, n_way*n_query]
        # z_rlc.permute(2, 0, 1) = [n_way*n_query, n_classes, n_dim]
        dists = euclidean_dist(z_query, z_lrc.permute(2, 0, 1))
        scores = -dists
        return scores
    
    def set_forward_reg(self, x,is_feature = False):
        z_support, _  = self.parse_feature(x,is_feature) # second result <z_query> is not required
        
        # Work on support vectors
        z_support   = z_support.contiguous()
        z_support_view     = z_support.view(self.n_way, self.n_support, -1 ) #the shape of z is [n_classes, n_support, n_dim]
        
        z_right = z_support_view.transpose(1,2).unsqueeze(1) # convert from [N, K, n_dim] to [N, 1, n_dim, K] (multiply K x n_dim matrix in [1] with every n_dim x K matrix in [2])
        z_mul =  torch.matmul(z_support_view, z_right) # [N, N, K, K]
        
        #mul_norms = torch.norm(z_mul, dim=(2,3)) # [N x N], take norm of all K x K matrices
        #mul_norms_sq = torch.mul(mul_norms, mul_norms)
        mul_norms = torch.mul(z_mul, z_mul)
        mul_norms_sq = torch.sum(mul_norms, dim=(2,3)) # [N x N], take SQUARED norm of all K x K matrices 
        
        #S_norms = torch.norm(z_support_view, dim=(1,2)) # [N,] norms of all support matrices: ||S_i||
        #S_norms_sq = torch.mul(S_norms, S_norms) # [N,] squared norms of all support matrices: ||S_i||^2
        S_norms = torch.mul(z_support_view, z_support_view)
        S_norms_sq = torch.sum(S_norms, dim=(1,2)) # [N, ]
        S_norms_mixed = torch.ger(S_norms_sq, S_norms_sq) # [N,N] multiplication combinations of norms: ||S_i||^2 * ||S_j||^2
        
        all_reg = torch.div(mul_norms_sq, S_norms_mixed)
        
        off_diag_ix = np.where(~np.eye(all_reg.size()[0],dtype=bool))
        ortho_reg = all_reg[off_diag_ix] # select off-diagonal elements
        
        total_reg = torch.sum(ortho_reg)
        
        return total_reg

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        reg_ortho = self.set_forward_reg(x)

        return self.loss_fn(scores, y_query) + self.lamb * reg_ortho


def euclidean_dist( x, y):
    # x: N x D
    # y: N x M x D
    n = x.size(0)
    m = y.size(1)
    d = x.size(1)
    assert d == y.size(2)

    x = x.unsqueeze(1).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
