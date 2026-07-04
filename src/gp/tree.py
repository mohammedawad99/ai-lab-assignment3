"""Expression trees for GP heuristics. Evaluated directly, no eval()."""

import math
from dataclasses import dataclass, field

from src.gp.primitives import apply_binary, apply_unary, clamp_value, terminal_value


@dataclass
class GPNode:
    kind: str  # "terminal", "const", "unary", "binary"
    value: str | float
    children: list["GPNode"] = field(default_factory=list)

    def evaluate(self, state) -> float:
        if self.kind == "terminal":
            return terminal_value(self.value, state)
        if self.kind == "const":
            return float(self.value)
        if self.kind == "unary":
            return clamp_value(apply_unary(self.value, self.children[0].evaluate(state)))
        if self.kind == "binary":
            left = self.children[0].evaluate(state)
            right = self.children[1].evaluate(state)
            return clamp_value(apply_binary(self.value, left, right))
        raise ValueError(f"unknown node kind '{self.kind}'")

    def copy(self) -> "GPNode":
        return GPNode(self.kind, self.value, [child.copy() for child in self.children])

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def depth(self) -> int:
        return 1 + max((child.depth() for child in self.children), default=0)

    def to_string(self) -> str:
        if self.kind == "terminal":
            return str(self.value)
        if self.kind == "const":
            return f"{self.value:g}"
        if self.kind == "unary":
            return f"{self.value}({self.children[0].to_string()})"
        left = self.children[0].to_string()
        right = self.children[1].to_string()
        if self.value in ("min", "max"):
            return f"{self.value}({left}, {right})"
        return f"({left} {self.value} {right})"


@dataclass
class GPTree:
    root: GPNode

    def evaluate(self, state) -> float:
        return self.root.evaluate(state)

    def copy(self) -> "GPTree":
        return GPTree(root=self.root.copy())

    def size(self) -> int:
        return self.root.size()

    def depth(self) -> int:
        return self.root.depth()

    def to_string(self) -> str:
        return self.root.to_string()

    def to_heuristic(self):
        """A safe heuristic callable: finite, clamped, never negative."""
        def heuristic(state) -> float:
            value = self.root.evaluate(state)
            if not math.isfinite(value):
                return 0.0
            return max(0.0, clamp_value(value))
        return heuristic
