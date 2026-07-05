"""Genetic Algorithm with an Island Model for CVRP.

Representation: a giant-tour chromosome, i.e. a permutation of all customer
ids without the depot. A simple capacity-aware scan splits the giant tour
into routes. Islands evolve separately and exchange their best individuals
in a ring every few generations.
"""

import time

import numpy as np

from src.cvrp.baseline import build_multistage_baseline
from src.cvrp.candidate_lists import build_candidate_lists
from src.cvrp.cost import solution_cost
from src.cvrp.distance import build_distance_matrix
from src.cvrp.local_search import improve_solution_advanced
from src.cvrp.model import CVRPInstance, CVRPSolution
from src.cvrp.solvers.neighborhood import clone_routes
from src.cvrp.solvers.result import CVRPSolverResult
from src.cvrp.validate import validate_solution

CONVERGENCE_EVERY = 10
PENALTY_COST = 1e9  # cost for chromosomes whose split is not feasible


# ---------- representation ----------

def customers_from_instance(instance: CVRPInstance) -> list[int]:
    return sorted(instance.customer_ids)


def split_giant_tour(instance: CVRPInstance, chromosome: list[int],
                     distance_matrix) -> CVRPSolution | None:
    """Greedy capacity-aware split: open a new route when the next customer
    would not fit. Returns None if the split needs too many vehicles."""
    depot = instance.depot_id
    routes = []
    route = [depot]
    load = 0.0
    for customer in chromosome:
        demand = instance.demands.get(customer, 0.0)
        if demand > instance.capacity:
            return None
        if load + demand > instance.capacity:
            route.append(depot)
            routes.append(route)
            route = [depot]
            load = 0.0
        route.append(customer)
        load += demand
    if len(route) > 1:
        route.append(depot)
        routes.append(route)
    if len(routes) > instance.vehicle_count:
        return None
    solution = CVRPSolution(routes=routes)
    solution.cost = solution_cost(solution, distance_matrix)
    return solution


def chromosome_from_solution(solution: CVRPSolution) -> list[int]:
    return [node for route in solution.routes for node in route if node != 0]


def random_chromosome(instance: CVRPInstance, rng) -> list[int]:
    customers = customers_from_instance(instance)
    return [int(c) for c in rng.permutation(customers)]


def ordered_crossover(parent1: list[int], parent2: list[int], rng) -> list[int]:
    """Classic OX: copy a slice from parent1, fill the rest in parent2 order."""
    n = len(parent1)
    if n < 2:
        return list(parent1)
    i = int(rng.integers(0, n - 1))
    j = int(rng.integers(i + 1, n))
    middle = parent1[i:j + 1]
    in_middle = set(middle)
    rest = [c for c in parent2 if c not in in_middle]
    return rest[:i] + middle + rest[i:]


def swap_mutation(chromosome: list[int], rng, mutation_rate: float = 0.1) -> list[int]:
    child = list(chromosome)
    if len(child) >= 2 and rng.random() < mutation_rate:
        a, b = rng.choice(len(child), size=2, replace=False)
        child[int(a)], child[int(b)] = child[int(b)], child[int(a)]
    return child


def inversion_mutation(chromosome: list[int], rng, mutation_rate: float = 0.1) -> list[int]:
    child = list(chromosome)
    if len(child) >= 2 and rng.random() < mutation_rate:
        i = int(rng.integers(0, len(child) - 1))
        j = int(rng.integers(i + 1, len(child)))
        child[i:j + 1] = child[i:j + 1][::-1]
    return child


# ---------- population, fitness, selection ----------

def make_initial_population(instance: CVRPInstance, distance_matrix,
                            population_size: int, rng) -> list[list[int]]:
    """One baseline chromosome plus random permutations."""
    baseline = build_multistage_baseline(instance).solution
    population = [chromosome_from_solution(baseline)]
    seen = {tuple(population[0])}
    attempts = 0
    while len(population) < population_size:
        chromosome = random_chromosome(instance, rng)
        key = tuple(chromosome)
        attempts += 1
        # avoid obvious duplicates, but never loop forever: with few customers
        # there may be fewer distinct permutations than population slots
        if key in seen and attempts < population_size * 10:
            continue
        seen.add(key)
        population.append(chromosome)
    return population


def evaluate_chromosome(instance: CVRPInstance, chromosome: list[int],
                        distance_matrix) -> float:
    solution = split_giant_tour(instance, chromosome, distance_matrix)
    if solution is None:
        return PENALTY_COST
    return solution.cost


def tournament_select(population: list[list[int]], costs: list[float], rng,
                      tournament_size: int = 3) -> list[int]:
    size = min(tournament_size, len(population))
    picked = rng.choice(len(population), size=size, replace=False)
    best_index = min((int(i) for i in picked), key=lambda i: costs[i])
    return list(population[best_index])


# ---------- island evolution and migration ----------

def evolve_island(instance: CVRPInstance, distance_matrix,
                  population: list[list[int]], generations: int, rng,
                  crossover_rate: float = 0.8, mutation_rate: float = 0.15,
                  elitism: int = 1):
    """Evolve one island. Returns (new_population, best_chromosome, best_cost)."""
    population = [list(c) for c in population]
    best_chromosome = None
    best_cost = float("inf")

    def note_best(costs):
        nonlocal best_chromosome, best_cost
        for chromosome, cost in zip(population, costs):
            if cost < best_cost:
                best_chromosome, best_cost = list(chromosome), cost

    for _ in range(generations):
        costs = [evaluate_chromosome(instance, c, distance_matrix) for c in population]
        note_best(costs)
        order = sorted(range(len(population)), key=lambda i: costs[i])
        new_population = [list(population[i]) for i in order[:elitism]]
        while len(new_population) < len(population):
            parent1 = tournament_select(population, costs, rng)
            parent2 = tournament_select(population, costs, rng)
            if rng.random() < crossover_rate:
                child = ordered_crossover(parent1, parent2, rng)
            else:
                child = list(parent1)
            child = swap_mutation(child, rng, mutation_rate)
            child = inversion_mutation(child, rng, mutation_rate)
            new_population.append(child)
        population = new_population

    costs = [evaluate_chromosome(instance, c, distance_matrix) for c in population]
    note_best(costs)
    return population, best_chromosome, best_cost


def migrate_ring(islands: list[list[list[int]]], distance_matrix,
                 instance: CVRPInstance, migrants: int = 1):
    """Each island sends its best migrants to the next island (ring),
    replacing the worst individuals there. Island sizes stay fixed."""
    if migrants <= 0 or len(islands) < 2:
        return islands

    all_costs = [
        [evaluate_chromosome(instance, c, distance_matrix) for c in island]
        for island in islands
    ]
    best_per_island = []
    for island, costs in zip(islands, all_costs):
        order = sorted(range(len(island)), key=lambda i: costs[i])
        best_per_island.append([list(island[i]) for i in order[:migrants]])

    new_islands = [[list(c) for c in island] for island in islands]
    for idx in range(len(islands)):
        target = (idx + 1) % len(islands)
        worst_order = sorted(range(len(islands[target])),
                             key=lambda i: all_costs[target][i], reverse=True)
        for m, migrant in enumerate(best_per_island[idx]):
            new_islands[target][worst_order[m]] = list(migrant)
    return new_islands


# ---------- full solver ----------

def run_cvrp_ga_island(instance: CVRPInstance, generations: int = 100,
                       population_size: int = 30, islands: int = 4,
                       seed: int = 42, timeout_sec: float = 10.0,
                       crossover_rate: float = 0.8, mutation_rate: float = 0.15,
                       migration_interval: int = 20,
                       migrants: int = 1,
                       advanced_local_search: bool = False,
                       local_search_every: int = 10,
                       advanced_max_passes: int = 1,
                       candidate_list_k: int | None = None) -> CVRPSolverResult:
    start_elapsed = time.perf_counter()
    start_cpu = time.process_time()

    distance_matrix = build_distance_matrix(instance)
    baseline = build_multistage_baseline(instance).solution
    initial_cost = baseline.cost
    best = CVRPSolution(routes=clone_routes(baseline.routes), cost=baseline.cost)

    # Stage 11-B: optional memetic step — polish the best solution every few
    # generations and reinject its chromosome into island 0
    neighbors = None
    if advanced_local_search and candidate_list_k is not None:
        neighbors = build_candidate_lists(distance_matrix, k=candidate_list_k)

    # one generator per island keeps islands independent and reproducible
    island_rngs = [np.random.default_rng(seed + i) for i in range(islands)]
    populations = [
        make_initial_population(instance, distance_matrix, population_size, island_rngs[i])
        for i in range(islands)
    ]

    convergence = []
    completed = 0

    def record(generation):
        convergence.append({
            "iteration": generation,
            "best_cost": best.cost,
            "current_cost": best.cost,
            "elapsed_time": time.perf_counter() - start_elapsed,
            "cpu_time": time.process_time() - start_cpu,
        })

    record(0)
    for gen in range(1, generations + 1):
        if time.perf_counter() - start_elapsed > timeout_sec:
            break
        completed = gen

        for i in range(islands):
            populations[i], island_best, island_cost = evolve_island(
                instance, distance_matrix, populations[i], 1, island_rngs[i],
                crossover_rate=crossover_rate, mutation_rate=mutation_rate,
            )
            if island_cost < best.cost - 1e-9:
                solution = split_giant_tour(instance, island_best, distance_matrix)
                if solution is not None:
                    best = solution

        if advanced_local_search and gen % local_search_every == 0 and \
                validate_solution(instance, best).feasible:
            polished = improve_solution_advanced(
                instance, best, distance_matrix, neighbors=neighbors,
                max_passes=advanced_max_passes)
            if polished.cost < best.cost - 1e-9 and \
                    validate_solution(instance, polished).feasible:
                best = polished
                # reinject the polished chromosome, replacing the newest child
                populations[0][-1] = chromosome_from_solution(best)

        if migration_interval > 0 and gen % migration_interval == 0:
            populations = migrate_ring(populations, distance_matrix, instance, migrants)

        if gen % CONVERGENCE_EVERY == 0:
            record(gen)

    if not convergence or convergence[-1]["iteration"] != completed:
        record(completed)

    check = validate_solution(instance, best)
    return CVRPSolverResult(
        algorithm="cvrp_ga_island",
        instance_name=instance.name,
        seed=seed,
        iterations=completed,
        best_solution=best,
        best_cost=best.cost,
        initial_cost=initial_cost,
        feasible=check.feasible,
        errors=check.errors,
        elapsed_time=time.perf_counter() - start_elapsed,
        cpu_time=time.process_time() - start_cpu,
        convergence=convergence,
    )
