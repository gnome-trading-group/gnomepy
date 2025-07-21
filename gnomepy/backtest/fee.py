from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import override


class FeeModel(ABC):
    @abstractmethod
    def calculate_fee(self, total_price: int, is_maker: bool):
        raise NotImplementedError

@dataclass
class StaticFeeModel(FeeModel):
    taker_fee: float
    maker_fee: float

    @override
    def calculate_fee(self, total_price: int, is_maker: bool):
        return (total_price * self.maker_fee) if is_maker else (total_price * self.taker_fee)
