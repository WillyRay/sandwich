from dataclasses import dataclass

@dataclass
class Run:
    id: int
    decay: float
    touchTransferFraction: float
    counts:  list[int]
    occupancies: list[int]
    cdffs: list[int]
    anyCps: list[int]

