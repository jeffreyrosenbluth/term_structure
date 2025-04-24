from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.core.parameter import Parameters

P = TypeVar("P", bound=Parameters)


class ShortRateModel(ABC, Generic[P]):
    """Base class for all short-rate models."""

    @abstractmethod
    def params(self) -> P: ...

    @abstractmethod
    def update_params(self, p: P) -> None: ...
