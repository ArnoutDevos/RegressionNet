# This code is authored by Arnout Devos
# Contains code from https://github.com/bertinetto/r2d2

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from torch import mm
from torch import transpose as t
#from torch import gesv
from torch import inverse as inv

class R2D2(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support):
        super(R2D2, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_S_norms = None
        self.keep_record = True
        
        self.n_augment = 1
        self.linsys = False
        self.init_adj_scale = 1e-4
        self.adj_base = 1
        self.lambda_base = 1
        self.init_lambda = 50
        self.learn_lambda = False
        
        
        # Custom params
        print("Initialized method R2D2 with regularization hyperparameter: {}".format(self.init_lambda))
        
        self.lambda_rr = LambdaLayer(self.learn_lambda, self.init_lambda, self.lambda_base)
        self.adjust = AdjustLayer(init_scale=self.init_adj_scale, base=self.adj_base)
        
    def set_forward(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        # Work on support vectors
        z_support   = z_support.contiguous()
        #z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(dim=1) #the shape of z is [n_data, n_dim], dim=1 is dimension to reduce (take mean)
        z_support_view     = z_support.view(self.n_way, self.n_support, -1 ) #the shape of z is [n_classes, n_support, n_dim]
        
        #print(z_support_view.shape)

        assert (z_support.size(0) == z_query.size(0))
        n_way, n_shot, n_query = z_support.size(0), z_support.size(1), z_query.size(1)
        
        self.n_augment = 1
        
        rr_type = 'woodbury'
        I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())
        
        y_inner = make_float_label(n_way, n_shot * self.n_augment) / np.sqrt(n_way * n_shot * self.n_augment)

        # add a column of ones for the bias
        
        #print(ones.shape)
        #print(z_support.shape[-1])
        z_support_view = z_support.view(self.n_way * self.n_support, -1 )
        ones = Variable(torch.unsqueeze(torch.ones(z_support_view.size(0)).cuda(), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((z_support_view, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        else:
            wb = self.rr_standard(torch.cat((z_support_view, ones), 1), n_way, n_shot, I, y_inner, self.linsys)
            
        #print(wb.shape)

        w = wb.narrow(dim=0, start=0, length=z_support.shape[-1])
        b = wb.narrow(dim=0, start=z_support.shape[-1], length=1)
        #print(z_query.shape)
        #print(w.shape)
        #print(b.shape)
        z_query_view = z_query.contiguous().view(-1, z_support.shape[-1] )
        out = mm(z_query_view, w) + b
        y_hat = self.adjust(out)
        scores = y_hat
        # print("%.3f  %.3f  %.3f" % (w.mean()*1e5, b.mean()*1e5, y_hat.max()))
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)
        #reg_ortho = self.set_forward_reg(x)
        #reg_angles = self.set_forward_principalangles(x)

        return self.loss_fn(scores, y_query) #+ self.lamb * reg_ortho
        #return self.loss_fn(scores, y_query) + self.lamb * reg_angles
        
    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot * self.n_augment)

        if not linsys:
            w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = torch.solve(v, A)
            w = mm(t_(x), w_)

        return w


def euclidean_dist( x, y):
    # x: N x D
    # y: N x M x D
    n = x.size(0)
    m = y.size(1)
    d = x.size(1)
    assert d == y.size(2)

    x = x.unsqueeze(1).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)



def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
    for i in range(n_way):
        label[n_samples * i:n_samples * (i + 1), i] = 1
    return to_variable(label)

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=False, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]).cuda()
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * (self.base ** self.l)

class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1e-4, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]).cuda())
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]).cuda())
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * (self.base ** self.scale) + self.base ** self.bias - 1
