from __future__ import annotations
import networkx as nx
import numpy as np
import torch

class KuramotoTorchResidual:
    def __init__(self,z_mean,z_scale):
        graph=nx.watts_strogatz_graph(64,6,.2,seed=0)
        self.A=torch.as_tensor(nx.to_numpy_array(graph,dtype=float)[:63],dtype=torch.float32)
        self.z_mean=torch.as_tensor(z_mean,dtype=torch.float32); self.z_scale=torch.as_tensor(z_scale,dtype=torch.float32)
    def __call__(self, y_standardized, q_physical):
        device=y_standardized.device; dtype=y_standardized.dtype
        z=y_standardized*self.z_scale.to(device=device,dtype=dtype)+self.z_mean.to(device=device,dtype=dtype)
        theta=torch.cat((z,torch.zeros((len(z),1),dtype=dtype,device=device)),dim=1)
        q=q_physical.to(device=device,dtype=dtype); omega=torch.cat((q,-q.sum(dim=1,keepdim=True)),dim=1)
        A=self.A.to(device=device,dtype=dtype)
        return omega[:,:63]-torch.sum(A[None,:,:]*torch.sin(theta[:,:63,None]-theta[:,None,:]),dim=2)
