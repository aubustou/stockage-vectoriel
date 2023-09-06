from __future__ import annotations
from dataclasses import dataclass, field, fields
import logging
import math
from typing import Literal
import uuid


@dataclass(kw_only=True)
class StorageBox:
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    name: str
    description: str = ""

    iops: int
    size: int

    is_occupied: bool = False

    def to_vector(self) -> list[float]:
        return [self.iops, self.size]


@dataclass(kw_only=True)
class StorageShard:
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    name: str
    description: str = ""
    mount_point: str
    pz: str
    hw_groups: list[str] = field(default_factory=list)

    iops: int
    iops_overcommit: float
    used_iops: int

    total_size: int
    used_size: int
    max_pct_size_used: float

    max_box_iops: int
    max_box_size: int

    state: Literal["active", "inactive"] = "active"

    @property
    def free_size(self) -> int:
        return math.ceil(
            self.total_size - self.used_size - self.total_size * self.max_pct_size_used
        )

    @property
    def total_iops(self) -> int:
        return math.ceil(self.iops * self.iops_overcommit)

    @property
    def free_iops(self) -> int:
        return self.total_iops - self.used_iops

    def to_vector(self) -> list[float]:
        return [self.free_iops, self.free_size]


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start generation")
