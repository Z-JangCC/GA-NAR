from __future__ import annotations
import torch
from torch import nn
from .baselines import MatchedActivation, MatchedSwiGLU

class GenericAblation(nn.Module):
    def __init__(self,input_dim,output_dim,projector,activation='relu',mode='capacity',width=384,
                 geometry_x=None,rho_target=None,pi_target=None,target_params=None,correction_width=None):
        super().__init__(); self.mode=mode; self.input_dim=input_dim
        self.register_buffer('P',torch.as_tensor(projector,dtype=torch.float32))
        self.anchor=(MatchedSwiGLU(input_dim,output_dim,hidden_dim=128,ff_width=256,blocks=4)
                     if activation=='swiglu' else MatchedActivation(input_dim,output_dim,activation,hidden_dim=128,ff_width=256,blocks=4))
        cdim=input_dim if mode=='capacity' else 3*input_dim+3
        if correction_width is not None: width=int(correction_width)
        elif target_params is not None: width=max(1,round((target_params-output_dim)/(cdim+output_dim+1)))
        self.correction=nn.Sequential(nn.Linear(cdim,width),nn.SiLU(),nn.Linear(width,output_dim))
        self.correction_width=width
        if geometry_x is not None:
            self.register_buffer('geometry_x',torch.as_tensor(geometry_x,dtype=torch.float32))
            self.register_buffer('rho_target',torch.as_tensor(rho_target,dtype=torch.float32))
            self.register_buffer('pi_target',torch.as_tensor(pi_target,dtype=torch.float32))
        for m in self.correction.modules():
            if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        self.correction[2].weight.data.zero_(); self.correction[2].bias.data.zero_()
    def _targets(self,x):
        # Fixed 4-nearest NRGD interpolation; no trainable geometry mechanism.
        dist=torch.cdist(x,self.geometry_x.to(x)); vals,idx=torch.topk(dist,k=min(4,len(self.geometry_x)),largest=False)
        w=1/(vals+1e-6); w=w/w.sum(dim=1,keepdim=True)
        rho=(w*self.rho_target.to(x)[idx]).sum(dim=1)
        pi=(w[:,:,None]*self.pi_target.to(x)[idx]).sum(dim=1)
        return rho,pi
    def features(self,x,rho=None,pi=None):
        if self.mode=='capacity': return x
        if rho is None or pi is None: rho,pi=self._targets(x)
        p=self.P.to(x); xr=x@p.T; xc=x@(torch.eye(self.input_dim,device=x.device,dtype=x.dtype)-p).T
        if rho is None: rho=torch.zeros((len(x),1),device=x.device,dtype=x.dtype)
        elif rho.ndim==1: rho=rho[:,None]
        if pi is None: pi=torch.zeros((len(x),2),device=x.device,dtype=x.dtype)
        return torch.cat((x,xr,xc,rho,pi),dim=-1)
    def forward(self,x,rho=None,pi=None,return_aux=False):
        pred=self.anchor(x)+self.correction(self.features(x,rho,pi))
        return (pred,None,None,None,None) if return_aux else pred
