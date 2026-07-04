"""Genetic operators for linear GEP genomes."""

from src.gep.genome import GEPGenome
from src.gep.symbols import random_head_symbol, random_tail_symbol


def point_mutation(genome: GEPGenome, rng, mutation_rate: float = 0.05) -> GEPGenome:
    """Mutate single genes. Head positions may become functions or terminals,
    tail positions stay terminals, so the genome remains valid."""
    genes = list(genome.genes)
    for i in range(len(genes)):
        if rng.random() < mutation_rate:
            if i < genome.head_length:
                genes[i] = random_head_symbol(rng)
            else:
                genes[i] = random_tail_symbol(rng)
    return GEPGenome(genes=genes, head_length=genome.head_length)


def _check_same_shape(parent1: GEPGenome, parent2: GEPGenome):
    if parent1.head_length != parent2.head_length:
        raise ValueError("parents must have the same head_length")


def one_point_crossover(parent1: GEPGenome, parent2: GEPGenome, rng) -> GEPGenome:
    _check_same_shape(parent1, parent2)
    point = int(rng.integers(1, parent1.total_length))
    genes = parent1.genes[:point] + parent2.genes[point:]
    return GEPGenome(genes=genes, head_length=parent1.head_length)


def two_point_crossover(parent1: GEPGenome, parent2: GEPGenome, rng) -> GEPGenome:
    _check_same_shape(parent1, parent2)
    a = int(rng.integers(0, parent1.total_length))
    b = int(rng.integers(0, parent1.total_length))
    i, j = min(a, b), max(a, b)
    if i == j:
        return parent1.copy()
    genes = parent1.genes[:i] + parent2.genes[i:j] + parent1.genes[j:]
    return GEPGenome(genes=genes, head_length=parent1.head_length)


def tournament_select(population: list[GEPGenome], fitnesses: list[float], rng,
                      tournament_size: int = 3) -> GEPGenome:
    """Sample a few genomes and return a copy of the fittest (higher is better)."""
    size = min(tournament_size, len(population))
    picked = rng.choice(len(population), size=size, replace=False)
    best_index = max((int(i) for i in picked), key=lambda i: fitnesses[i])
    return population[best_index].copy()
