from __future__ import annotations

from .power_flow_dataset import build_power_flow_dataset
from .synthetic_dataset import build_synthetic_dataset


class DatasetProvider:
    """Public dataset entry point; downstream modules do not call generators directly."""

    def __init__(self, benchmark: str, system, **kwargs) -> None:
        self.benchmark = benchmark
        self.system = system
        self.kwargs = kwargs

    def build(self):
        if self.benchmark == "synthetic":
            return build_synthetic_dataset(self.system, **self.kwargs)
        if self.benchmark == "ieee118":
            return build_power_flow_dataset(self.system, **self.kwargs)
        raise KeyError(self.benchmark)

