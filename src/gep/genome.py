"""Linear GEP genome with head/tail structure."""

from dataclasses import dataclass

from src.gep.symbols import MAX_ARITY, is_function, is_terminal


@dataclass
class GEPGenome:
    genes: list[str]
    head_length: int

    @property
    def tail_length(self) -> int:
        # standard GEP rule: t = h * (max_arity - 1) + 1
        return self.head_length * (MAX_ARITY - 1) + 1

    @property
    def total_length(self) -> int:
        return self.head_length + self.tail_length

    def __post_init__(self):
        if self.head_length < 1:
            raise ValueError(f"head_length must be >= 1, got {self.head_length}")
        if len(self.genes) != self.total_length:
            raise ValueError(
                f"genome needs {self.total_length} genes "
                f"(head {self.head_length} + tail {self.tail_length}), got {len(self.genes)}"
            )
        for symbol in self.genes[:self.head_length]:
            if not (is_function(symbol) or is_terminal(symbol)):
                raise ValueError(f"unknown head symbol '{symbol}'")
        for symbol in self.genes[self.head_length:]:
            if not is_terminal(symbol):
                raise ValueError(f"tail symbol '{symbol}' must be a terminal")

    def copy(self) -> "GEPGenome":
        return GEPGenome(genes=list(self.genes), head_length=self.head_length)

    def to_string(self) -> str:
        return " ".join(self.genes)
