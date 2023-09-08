from __future__ import annotations
from dataclasses import dataclass, field, fields
from functools import cached_property
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
        return math.ceil(self.total_size * self.max_pct_size_used - self.used_size)

    @property
    def total_iops(self) -> int:
        return math.ceil(self.iops * self.iops_overcommit)

    @property
    def free_iops(self) -> int:
        return self.total_iops - self.used_iops

    def to_vector(self) -> list[float]:
        return [self.free_iops, self.free_size]

    @cached_property
    def dimension_space(self) -> DimensionSpace:
        return DimensionSpace(iops=(0, self.free_iops), size=(0, self.free_size))


@dataclass(kw_only=True)
class DimensionSpace:
    iops: tuple[int, int] = field(metadata={"unit": "iops"})
    size: tuple[int, int] = field(metadata={"unit": "GB"})

    def to_vector(self) -> list[tuple[int, int]]:
        return [getattr(self, x.name) for x in fields(self) if x.metadata.get("unit")]


def create_boxes(
    storage_shards: list[StorageShard],
) -> list[StorageBox]:
    boxes: list[StorageBox] = []
    for shard in storage_shards:
        boxes.extend(create_boxes_in_shard(shard))

    return boxes


IOPS_CUTOFF = 10
SIZE_CUTOFF = 10


def create_boxes_in_shard(storage_shard: StorageShard) -> list[StorageBox]:
    """Recursively create boxes for all dimensions in a shard."""
    boxes: list[StorageBox] = []

    dimensions = storage_shard.dimension_space.to_vector()
    iops_min, iops_max = dimensions[0]
    size_min, size_max = dimensions[1]
    logging.info("Creating boxes for %s", storage_shard.name)
    logging.debug(
        "Dimensions for %s: iops: %s-%s, size: %s-%s",
        storage_shard.name,
        iops_min,
        iops_max,
        size_min,
        size_max,
    )

    iops = iops_min

    while iops < iops_max:
        iops = math.ceil((iops_max - iops) / 2) + iops
        if (new_iops := iops_max - iops) < IOPS_CUTOFF:
            break

        size = size_min
        while size < size_max:
            size = math.ceil((size_max - size) / 2) + size

            if (new_size := size_max - size) < SIZE_CUTOFF:
                break
            logging.debug(
                "Creating box for %s with iops %s and size %s",
                storage_shard.name,
                new_iops,
                new_size,
            )
            boxes.append(
                StorageBox(
                    name=f"{storage_shard.name}-{iops}-{size}",
                    description=f"Box for {storage_shard.name} with {iops} iops and {size} GB",
                    iops=new_iops,
                    size=new_size,
                )
            )

    return boxes


def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start generation")

    logging.info("Create storage shards")
    storage_shards = [
        StorageShard(
            name="shard1",
            description="First shard",
            mount_point="/mnt/shard1",
            pz="pz1",
            hw_groups=["hw1", "hw2"],
            iops=1000,
            iops_overcommit=1.5,
            used_iops=500,
            total_size=1000,
            used_size=500,
            max_pct_size_used=0.8,
            max_box_iops=100,
            max_box_size=100,
        ),
        StorageShard(
            name="shard2",
            description="Second shard",
            mount_point="/mnt/shard2",
            pz="pz2",
            hw_groups=["hw1", "hw3"],
            iops=1000,
            iops_overcommit=1.5,
            used_iops=500,
            total_size=1000,
            used_size=500,
            max_pct_size_used=0.8,
            max_box_iops=100,
            max_box_size=100,
        ),
    ]

    logging.info("Create boxes")
    boxes = create_boxes(storage_shards)

    logging.info("Boxes: %s", len(boxes))
    for box in boxes:
        logging.info("\tBox: %s", box.name)


if __name__ == "__main__":
    main()
