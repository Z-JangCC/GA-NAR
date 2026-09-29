from __future__ import annotations
import numpy as np
import torch
from pypower.api import case118
from pypower.makeYbus import makeYbus

class PowerFlowTorchResidual:
    def __init__(self,z_mean,z_scale):
        c=case118(); self.base=float(c['baseMVA']); bus=c['bus'].copy(); gen=c['gen'].copy(); branch=c['branch'].copy()
        bus[:,0]-=1; gen[:,0]-=1; branch[:,:2]-=1
        self.bus=bus; self.slack=int(np.where(bus[:,1]==3)[0][0]); self.pq=np.where(bus[:,1]==1)[0]
        self.non_slack=np.array([i for i in range(118) if i!=self.slack]); self.load=np.where((bus[:,1]!=3)&((abs(bus[:,2])+abs(bus[:,3]))>0))[0]
        Y,_,_=makeYbus(self.base,bus,branch); Y=np.asarray(Y.toarray())
        # Keep the evaluator entirely real-valued.  This is algebraically
        # identical to S = V * conj(YV), but avoids CUDA's complex-exp
        # Jiterator/NVRTC dependency.
        self.G=torch.as_tensor(Y.real,dtype=torch.float64)
        self.B=torch.as_tensor(Y.imag,dtype=torch.float64)
        Pg=np.zeros(118); Qg=np.zeros(118)
        for row in gen: Pg[int(row[0])]+=row[1]/self.base; Qg[int(row[0])]+=row[2]/self.base
        self.Pg=torch.as_tensor(Pg,dtype=torch.float64); self.Qg=torch.as_tensor(Qg,dtype=torch.float64)
        self.Pd=torch.as_tensor(bus[:,2]/self.base,dtype=torch.float64); self.Qd=torch.as_tensor(bus[:,3]/self.base,dtype=torch.float64)
        self.V0=torch.as_tensor(bus[:,7],dtype=torch.float64); self.z_mean=torch.as_tensor(z_mean,dtype=torch.float64); self.z_scale=torch.as_tensor(z_scale,dtype=torch.float64)
    def __call__(self,y,q):
        dev=y.device; dtype=y.dtype; z=y*self.z_scale.to(dev,dtype)+self.z_mean.to(dev,dtype); q=q.to(dev,dtype)
        th=torch.zeros((len(z),118),device=dev,dtype=dtype); th[:,self.non_slack]=z[:,:len(self.non_slack)]
        V=self.V0.to(dev,dtype)[None,:].repeat(len(z),1); V[:,self.pq]=z[:,len(self.non_slack):]
        vr=V*torch.cos(th); vi=V*torch.sin(th)
        G=self.G.to(dev,dtype); B=self.B.to(dev,dtype)
        ir=torch.einsum('ij,bj->bi',G,vr)-torch.einsum('ij,bj->bi',B,vi)
        ii=torch.einsum('ij,bj->bi',B,vr)+torch.einsum('ij,bj->bi',G,vi)
        P=vr*ir+vi*ii; Q=vi*ir-vr*ii
        scale=torch.ones_like(V); scale[:,self.load]=q
        Psp=self.Pg.to(dev,dtype)-scale*self.Pd.to(dev,dtype); Qsp=self.Qg.to(dev,dtype)-scale*self.Qd.to(dev,dtype)
        return torch.cat((Psp[:,self.non_slack]-P[:,self.non_slack],Qsp[:,self.pq]-Q[:,self.pq]),dim=1)
