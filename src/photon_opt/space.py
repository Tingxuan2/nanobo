from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Parameter:
    """One design parameter with physical bounds, e.g. a disk radius in nm."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Parameter '{self.name}': low ({self.low}) must be < high ({self.high})"
            )

    @property
    def span(self) -> float:
        """Width of the allowed range."""
        return self.high - self.low


@dataclass(frozen=True)
class ParameterSpace:
    """An ordered collection of parameters. Column i of any tensor = parameter i."""

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        if len(self.parameters) == 0:
            raise ValueError("ParameterSpace needs at least one parameter")
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names: {names}")

    @property
    def dim(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    @property
    def names(self) -> list[str]:
        """Parameter names, in order."""
        return [p.name for p in self.parameters]
    
    @property
    def lows(self) -> torch.Tensor:
        """Lower bounds, shape (d,)."""
        return torch.tensor([p.low for p in self.parameters], dtype=torch.float64)

    @property
    def highs(self) -> torch.Tensor:
        """Upper bounds, shape (d,)."""
        return torch.tensor([p.high for p in self.parameters], dtype=torch.float64)

    def to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """Physical values (..., d) -> normalized values in [0, 1]."""
        return (x - self.lows) / (self.highs - self.lows)

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        """Normalized values (..., d) in [0, 1] -> physical values."""
        return self.lows + u * (self.highs - self.lows)