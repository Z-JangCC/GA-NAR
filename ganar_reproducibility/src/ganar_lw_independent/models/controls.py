from __future__ import annotations

from .layerwise_ganar import LayerwiseGANAR


def create_control(name: str, **kwargs):
    modes = {
        "ganar_lw": "layerwise",
        "ganar_uniform_beta": "uniform_beta",
        "ganar_no_geometry": "no_geometry",
        "ganar_prebackbone": "prebackbone",
    }
    if name not in modes:
        raise KeyError(name)
    return LayerwiseGANAR(mode=modes[name], **kwargs)
