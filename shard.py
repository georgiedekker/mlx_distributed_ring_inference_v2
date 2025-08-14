"""
Shard utilities for pipeline parallelism.

This module defines a simple ``Shard`` dataclass used to describe
sub‑ranges of model layers. It includes convenience methods for
detecting overlaps and computing the number of layers in a shard. While
the DeepSeek model used by this example does not require manual
instantiation of ``Shard`` objects—the built‑in ``pipeline()`` method
handles sharding automatically—this class is retained for compatibility
with code that expects a ``shard`` attribute on model arguments.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Shard:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int

    def __hash__(self) -> int:
        return hash((self.model_id, self.start_layer, self.end_layer, self.n_layers))

    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        return self.end_layer == self.n_layers - 1

    def get_layer_count(self) -> int:
        return self.end_layer - self.start_layer + 1

    def to_dict(self) -> Dict[str, int]:
        return {
            "model_id": self.model_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "n_layers": self.n_layers,
        }

    @staticmethod
    def from_dict(data: Dict) -> "Shard":
        return Shard(**data)

    def overlaps(self, other: "Shard") -> bool:
        return shards_overlap(self, other)


def shards_overlap(shard1: Shard, shard2: Shard) -> bool:
    return (
        shard1.model_id == shard2.model_id
        and max(shard1.start_layer, shard2.start_layer)
        <= min(shard1.end_layer, shard2.end_layer)
    )