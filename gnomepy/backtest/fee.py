from dataclasses import dataclass


@dataclass
class FeeModel:
    taker_fee: float
    maker_fee: float
