"""optimizers. this update 'p.data' in place using 'p.grad'"""

import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params=list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad=None


    def step(self):
        raise NotImplementedError



class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay=lr, momentum, weight_decay
        self.velocity=[np.zeros_like(p.data) for p in self.params]


    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue

            g=p.grad+self.weight_decay*p.data if self.weight_decay else p.grad

            if self.momentum:
                v*=self.momentum
                v+=g
                g=v

            p.data-=(self.lr*g).astype(p.data.dtype)



class Adam(Optimizer):
    """Adam with optional L2 weight decay (same update rule as torch.optim.Adam)"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.betas, self.eps, self.weight_decay=lr,betas,eps,weight_decay
        self.m=[np.zeros_like(p.data) for p in self.params]
        self.v=[np.zeros_like(p.data) for p in self.params]
        self.t=0


    def step(self):
        self.t +=1
        b1, b2=self.betas
        bias1=1-b1**self.t
        bias2=1-b2**self.t


        for p,m,v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue

            g=p.grad+self.weight_decay* p.data if self.weight_decay else p.grad
            m*=b1
            m+=(1-b1)*g
            v*=b2
            v+=(1-b2)*g*g
            denom=np.sqrt(v)/np.sqrt(bias2)+self.eps
            p.data-=((self.lr / bias1) * m/denom).astype(p.data.dtype)