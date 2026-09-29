from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .data import load_independent_dataset
from .models.baselines import MatchedActivation, MatchedSwiGLU
from .models.layerwise_ganar import LayerwiseGANAR
from .models.ablation import GenericAblation


def _model(system, name, seed):
    data = load_independent_dataset(system)
    d, m = data.x.shape[1], data.y.shape[1]
    if name.startswith("capacity_") or name.startswith("generic_geometry_"):
        activation=name.split('_')[-1]; mode='capacity' if name.startswith('capacity_') else 'generic'
        ckpath=Path("local_training_results")/system/name/str(seed)/"checkpoint.pt"
        ckraw=torch.load(ckpath,map_location="cpu",weights_only=False).get("state_dict",{})
        correction_width=int(ckraw["correction.0.weight"].shape[0])
        reference=LayerwiseGANAR(d,m,data.projector,activation='silu' if activation=='swiglu' else activation,
                                 block_kind='swiglu' if activation=='swiglu' else 'activation')
        target=sum(p.numel() for p in reference.delta_output.parameters())+sum(p.numel() for p in reference.input_skip.parameters())
        model=GenericAblation(d,m,data.projector,activation,mode,geometry_x=data.geometry_x,
                              rho_target=data.rho,pi_target=data.pi,target_params=target,
                              correction_width=correction_width)
        warm_path=Path("local_training_results")/system/("pm_"+activation)/str(seed)/"checkpoint.pt"
        if warm_path.exists():
            warm=torch.load(warm_path,map_location="cpu",weights_only=False).get("state_dict",{})
            model.anchor.load_state_dict(warm,strict=False)
        model.load_state_dict(ckraw,strict=True)
        return data,model.double().eval()
    else:
        paired_activation = name.startswith("ganar_") and name[6:] in {"relu","gelu","silu","mish","softplus","swiglu"}
    if 'paired_activation' in locals() and paired_activation:
        activation = name[6:]
        if activation == "swiglu":
            # Match run_one.build_model: the final paired runs use the
            # independent stem (anchor_features=False).  The default
            # LayerwiseGANAR constructor is True, and silently selecting that
            # variant changes the forward graph even though the checkpoint
            # state_dict keys still load.
            model = LayerwiseGANAR(d, m, data.projector, activation="silu",
                                   block_kind="swiglu", anchor_features=False)
        else:
            model = LayerwiseGANAR(d, m, data.projector, activation=activation,
                                   anchor_features=False)
        model.anchor_only = True
        warm_path = Path("local_training_results") / system / ("pm_" + activation) / str(seed) / "checkpoint.pt"
        if warm_path.exists():
            warm = torch.load(warm_path, map_location="cpu", weights_only=False).get("state_dict", {})
            model.anchor.load_state_dict(warm, strict=False)
    elif name == "ganar_lw":
        model = LayerwiseGANAR(d, m, data.projector, activation="silu",
                               anchor_features=False)
        # The current formal GA-NAR-LW checkpoint uses the paired anchor mode
        # on every system, not only on Kuramoto.
        model.anchor_only = True
    elif name.startswith("pm_") and name[3:] in MatchedActivation.ACTIVATIONS:
        model = MatchedActivation(d, m, name[3:], ff_width=256)
    elif name == "pm_silu":
        model = MatchedActivation(d, m, "silu", ff_width=256)
    elif name == "pm_mish":
        model = MatchedActivation(d, m, "mish", ff_width=256)
    else:
        model = MatchedSwiGLU(d, m, ff_width=256)
    ck = torch.load(Path("local_training_results") / system / name / str(seed) / "checkpoint.pt",
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=False)
    return data, model.double().eval()


def _kuramoto_residual(z, q):
    graph = nx.watts_strogatz_graph(64, 6, .2, seed=0)
    A = nx.to_numpy_array(graph, dtype=float)
    theta = np.r_[z, 0.0]
    omega = np.r_[q, -np.sum(q)]
    return omega[:63] - np.sum(A[:63] * np.sin(theta[:63, None] - theta[None, :]), axis=1)


def _powerflow_residual(z, q):
    # Independent AC power-flow residual using PYPOWER case118 primitives.
    from pypower.api import case118
    from pypower.makeYbus import makeYbus
    case = case118(); base = float(case["baseMVA"])
    bus = case["bus"].copy(); gen = case["gen"].copy(); branch = case["branch"].copy()
    bus[:, 0] -= 1; gen[:, 0] -= 1; branch[:, :2] -= 1
    slack = int(np.where(bus[:, 1] == 3)[0][0])
    pq = np.where(bus[:, 1] == 1)[0]
    non_slack = np.array([i for i in range(118) if i != slack])
    load = np.where((bus[:, 1] != 3) & ((np.abs(bus[:, 2]) + np.abs(bus[:, 3])) > 0))[0]
    Y, _, _ = makeYbus(base, bus, branch)
    Pg = np.zeros(118); Qg = np.zeros(118)
    for row in gen:
        Pg[int(row[0])] += row[1] / base; Qg[int(row[0])] += row[2] / base
    Pd0 = bus[:, 2] / base; Qd0 = bus[:, 3] / base
    theta = np.zeros(118); theta[non_slack] = z[:len(non_slack)]
    V = bus[:, 7].copy(); V[pq] = z[len(non_slack):]
    S = V * np.exp(1j * theta) * np.conj(Y @ (V * np.exp(1j * theta)))
    scale = np.ones(118); scale[load] = q
    Psp = Pg - scale * Pd0; Qsp = Qg - scale * Qd0
    return np.r_[Psp[non_slack] - S.real[non_slack], Qsp[pq] - S.imag[pq]]


class _ExactHeatFEM:
    """Standalone copy of the registered degree-5 triangular FEM residual."""

    def __init__(self, grid=24):
        self.grid = grid; self.n = grid - 1
        self.state_dimension = self.n * self.n
        self.modes = [(a, b) for a in range(1, 5) for b in range(1, 5)]
        self.quad = np.array([[1/3,1/3,.225], [.470142064105115,.470142064105115,.132394152788506],
                              [.470142064105115,.059715871789770,.132394152788506],
                              [.059715871789770,.470142064105115,.132394152788506],
                              [.101286507323456,.101286507323456,.125939180544827],
                              [.101286507323456,.797426985353087,.125939180544827],
                              [.797426985353087,.101286507323456,.125939180544827]])
        self._build_geometry()

    def _node(self, i, j): return i*(self.grid+1)+j

    def _build_geometry(self):
        h=1/self.grid; self.full_to_int={self._node(i,j):(i-1)*self.n+(j-1)
            for i in range(1,self.grid) for j in range(1,self.grid)}
        elements=[]
        for i in range(self.grid):
            for j in range(self.grid):
                elements += [(np.array([self._node(i,j),self._node(i+1,j),self._node(i+1,j+1)]),
                              (i,j)), (np.array([self._node(i,j),self._node(i+1,j+1),self._node(i,j+1)]),
                              (i,j))]
        self.elem_int=[]; self.elem_grads=[]; self.quad_xy=[]; self.quad_N=[]; self.quad_w=[]
        for nodes,_ in elements:
            ij=np.array([(u//(self.grid+1),u%(self.grid+1)) for u in nodes]); xy=ij*h
            x1,x2,x3=xy; B=np.array([[x2[0]-x1[0],x3[0]-x1[0]],[x2[1]-x1[1],x3[1]-x1[1]]])
            det=np.linalg.det(B); invB=np.linalg.inv(B)
            grads=np.c_[invB.T@np.array([-1.,-1.]),invB.T@np.array([1.,0.]),invB.T@np.array([0.,1.])].T
            ints=np.array([self.full_to_int.get(int(u),-1) for u in nodes]); self.elem_int.append(ints); self.elem_grads.append(grads)
            pts=[]; ns=[]; ws=[]
            for a,b,w in self.quad:
                pts.append(x1+B@np.array([a,b])); ns.append(np.array([1-a-b,a,b])); ws.append(.5*w*abs(det))
            self.quad_xy.append(pts); self.quad_N.append(ns); self.quad_w.append(ws)
        self.elem_int=np.asarray(self.elem_int); self.elem_grads=np.asarray(self.elem_grads)
        self.quad_xy=np.asarray(self.quad_xy); self.quad_N=np.asarray(self.quad_N); self.quad_w=np.asarray(self.quad_w)

    def residual(self,z,q):
        z=np.asarray(z); q=np.asarray(q); tv=np.zeros((len(self.elem_int),3)); mask=self.elem_int>=0; tv[mask]=z[self.elem_int[mask]]
        gradT=np.einsum('eai,ea->ei',self.elem_grads,tv); Tg=np.einsum('ega,ea->eg',self.quad_N,tv)
        x=self.quad_xy; basis=np.stack([np.sin(a*np.pi*x[...,0])*np.sin(b*np.pi*x[...,1]) for a,b in self.modes],axis=-1)
        source=10*(1+.05*np.einsum('egk,k->eg',basis,q)); gradterm=np.einsum('eai,ei->ea',self.elem_grads,gradT)
        local=self.quad_w[...,None]*((1+Tg*Tg)[...,None]*gradterm[:,None,:]+.1*Tg[...,None]**3*self.quad_N-source[...,None]*self.quad_N)
        out=np.zeros(self.state_dimension)
        for a in range(3):
            good=self.elem_int[:,a]>=0; np.add.at(out,self.elem_int[good,a],local[good,:,a].sum(1))
        return out


_EXACT_HEAT = _ExactHeatFEM()


def _heat_residual(z, q):
    return _EXACT_HEAT.residual(z, q)


def _inverse_output(system, data):
    raw = np.load(Path("data_store/processed") / system / "dataset.npz")
    return raw["z_mean"], raw["z_scale"]


def evaluate(system, model_name, seed):
    data, model = _model(system, model_name, seed)
    raw = np.load(Path("data_store/processed") / system / "dataset.npz")
    x = torch.as_tensor(data.x[data.test], dtype=torch.float64)
    with torch.no_grad(): pred = model(x).cpu().numpy()
    z = pred * raw["z_scale"] + raw["z_mean"]
    q = data.x[data.test] * raw["q_scale"] + raw["q_mean"]
    residual_fn = {"kuramoto64": _kuramoto_residual,
                   "ieee118_acpf": _powerflow_residual,
                   "nonlinear_heat_fem": _heat_residual}[system]
    residuals = np.asarray([residual_fn(zz, qq) for zz, qq in zip(z, q)])
    norms = np.linalg.norm(residuals, axis=1) / np.sqrt(residuals.shape[1])
    high = np.linalg.norm(q, axis=1) >= np.quantile(np.linalg.norm(q, axis=1), .75)
    return {"system": system, "model": model_name, "seed": seed,
            "residual_rms": float(np.sqrt(np.mean(residuals**2))),
            "residual_norm_mean": float(norms.mean()),
            "residual_norm_p95": float(np.quantile(norms, .95)),
            "residual_norm_max": float(norms.max()),
            "high_nonlinearity_residual_mean": float(norms[high].mean()),
            "admissible_fraction": float(np.mean(np.isfinite(residuals).all(axis=1)))}


def main():
    settings={"ieee118_acpf":["relu","gelu","swiglu"],"kuramoto64":["relu","gelu"],"nonlinear_heat_fem":["relu","softplus"]}
    systems={s:[f"pm_{a}" for a in acts]+[f"capacity_{a}" for a in acts]+[f"generic_geometry_{a}" for a in acts]+[f"ganar_{a}" for a in acts] for s,acts in settings.items()}
    rows = []
    for system, models in systems.items():
        for model in models:
            for seed in (0, 1, 2):
                try:
                    rows.append(evaluate(system, model, seed))
                except Exception as exc:
                    rows.append({"system": system, "model": model, "seed": seed,
                                 "status": "failed", "error": repr(exc)})
    out = Path("local_training_results/physics_satisfaction.json")
    out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
