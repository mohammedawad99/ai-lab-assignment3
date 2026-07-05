"""Random GEP genome generation."""

from src.gep.genome import GEPGenome
from src.gep.symbols import MAX_ARITY, TERMINALS, random_head_symbol, random_tail_symbol


def random_genome(rng, head_length: int = 6, terminals=TERMINALS) -> GEPGenome:
    tail_length = head_length * (MAX_ARITY - 1) + 1
    head = [random_head_symbol(rng, terminals) for _ in range(head_length)]
    tail = [random_tail_symbol(rng, terminals) for _ in range(tail_length)]
    return GEPGenome(genes=head + tail, head_length=head_length)


def make_initial_population(rng, population_size: int = 20, head_length: int = 6,
                            terminals=TERMINALS) -> list[GEPGenome]:
    return [random_genome(rng, head_length, terminals)
            for _ in range(population_size)]
