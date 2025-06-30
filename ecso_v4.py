import ioh
import numpy as np
import random
import os
import argparse

eval_count = 0
budget = 10000


class BudgetExceededException(Exception):
    pass


def evaluate(f, x):
    global eval_count
    if f.state.evaluations >= budget or eval_count >= budget:
        raise BudgetExceededException(f"⚠️ Evaluation budget of {budget} exceeded.")
    eval_count += 1
    return f(x)


def compute_violation_total(f,x):
    evaluate(f,x)
    
    total_violation = 0
    for i in range(0,f.constraints.n()):
        total_violation += f.constraints[i].violation()
    if is_feasible(f,x):
        return total_violation
    else:
        return -total_violation

def is_feasible(f, x):
    for i in range(0,f.constraints.n()):
        if f.constraints[i].compute_violation(x):
            return False
    return True



def smart_initialize(f, pop_size=100, max_ones=10):
    """
    Initializes a population of feasible binary solutions.
    Each solution greedily adds up to `max_ones` bits set to 1, preserving feasibility.
    Returns exactly `pop_size` feasible solutions.
    """

    feasible_population = []
    n = f.meta_data.n_variables
    max_ones = int(0.4 * n)
    max_attempts = pop_size * 10  # safety bound to prevent infinite loop
    attempts = 0

    while len(feasible_population) < pop_size and attempts < max_attempts:
        x = np.zeros(n, dtype=int)
        indices = list(range(n))
        random.shuffle(indices)

        current = x.copy()
        for i in range(min(max_ones, n)):
            current[indices[i]] = 1
            if not is_feasible(f, current):
                current[indices[i]] = 0  # revert if infeasible

        if is_feasible(f, current):
            feasible_population.append(current.copy())

        attempts += 1

    if len(feasible_population) < pop_size:
        print(f"⚠️ Warning: Only generated {len(feasible_population)} feasible individuals out of requested {pop_size}")

    return feasible_population



def init_population_greedy_ranked_extensions(f, pop_size=100, top_k=5):
    """
    Greedy + objective-aware initialization.
    Builds each individual step-by-step by adding bits that improve the objective,
    picking randomly among the top-k feasible single-bit flips.
    """
    n = f.meta_data.n_variables
    population = []

    while len(population) < pop_size:
        x = np.zeros(n, dtype=int)
        start_idx = random.randint(0, n - 1)
        x[start_idx] = 1
        if not is_feasible(f, x):
            x[start_idx] = 0 

        while True:
            candidates = []
            for i in range(n):
                if x[i] == 0:
                    x_try = x.copy()
                    x_try[i] = 1
                    if is_feasible(f, x_try):
                        obj = f(x_try)
                        candidates.append((obj, x_try))

            if not candidates:
                break  
                
            candidates.sort(reverse=True, key=lambda t: t[0])
            top_candidates = candidates[:min(top_k, len(candidates))]
            _, chosen = random.choice(top_candidates)
            x = chosen  

        population.append(x.copy())
        print("Current length of population = ", len(population))

    return population


def init_population_greedy_ranked_extensions_sampled(f, pop_size=100, top_k=5):
    sample_size=f.meta_data.n_variables // 4
    """
    Greedy + objective-aware initialization with sampling to reduce evaluations.
    Builds each individual by greedily adding bits that improve the objective,
    sampling a subset of zero-bits to evaluate at each step.
    """
    n = f.meta_data.n_variables
    population = []

    while len(population) < pop_size:
        x = np.zeros(n, dtype=int)
        start_idx = random.randint(0, n - 1)
        x[start_idx] = 1
        if not is_feasible(f, x):
            x[start_idx] = 0

        while True:
            candidates = []
            zero_indices = np.where(x == 0)[0]

            if len(zero_indices) == 0:
                break

            # Sample a subset of indices to try
            sample_indices = random.sample(
                list(zero_indices), min(sample_size, len(zero_indices))
            )

            for i in sample_indices:
                x_try = x.copy()
                x_try[i] = 1
                if is_feasible(f, x_try):
                    obj = f(x_try)  # objective evaluation
                    candidates.append((obj, x_try))

            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda t: t[0])
            top_candidates = candidates[:min(top_k, len(candidates))]
            _, chosen = random.choice(top_candidates)
            x = chosen

        population.append(x.copy())
        print("Current length of population =", len(population))

    return population


def init_fast_population_greedy_ranked_extensions(f, pop_size=100, top_k=5):
    """
    Greedy initialization using dynamically selected top 10% of remaining 0-bit positions.
    Bit scores are precomputed once, but filtered dynamically each step.
    """
    n = f.meta_data.n_variables
    population = []

    # Precompute bit-wise scores
    bit_scores = []
    for i in range(n):
        x = np.zeros(n, dtype=int)
        x[i] = 1
        score = evaluate(f,x) if is_feasible(f, x) else float('-inf')
        bit_scores.append((score, i))

    # Sort high to low
    bit_scores.sort(reverse=True, key=lambda t: t[0])

    while len(population) < pop_size:
        x = np.zeros(n, dtype=int)
        tries = 0
        max_tries = 10**4

        # Randomly select one good starting bit (feasible)
        for _, idx in bit_scores:
            x[idx] = 1
            if is_feasible(f, x):
                break
            x[idx] = 0

        # Iteratively extend x
        while tries < max_tries:
            # Identify remaining 0-positions
            remaining_indices = [i for i in range(n) if x[i] == 0]
#             print(remaining_indices)

            # Extract their scores from precomputed list
            filtered_scores = [(score, idx) for score, idx in bit_scores if idx in remaining_indices]

            if not filtered_scores:
                break

            # Take top 50% of remaining indices
            top_count = max(1, int(0.1 * len(filtered_scores)))
            top_candidates_raw = filtered_scores[:top_count]

            # Try flipping each and collect feasible ones
            candidates = []
            for _, idx in top_candidates_raw:
                x_try = x.copy()
                x_try[idx] = 1
                if is_feasible(f, x_try):
                    obj = f(x_try)
                    candidates.append((obj, x_try))
#                     print(len(candidates))

            if not candidates:
                break

            # Choose best from top_k
            candidates.sort(reverse=True, key=lambda t: t[0])
            top_k_candidates = candidates[:min(top_k, len(candidates))]
            _, chosen = random.choice(top_k_candidates)
            x = chosen
#             print(x)
            tries += 1
    
        if is_feasible(f,x):
            print(x)
            population.append(x.copy())
        print("Current length of population =", len(population))

    return population



def init_population_fast(f, pop_size=100, top_k=5, candidate_sample_size=10):
    """
    Faster version of greedy + objective-aware initialization.
    Instead of checking all 0-bits, sample a few candidates at each step.
    """
    n = f.meta_data.n_variables
    population = []

    while len(population) < pop_size:
        x = np.zeros(n, dtype=int)

        # Step 1: Try to find a feasible initial bit
        shuffled_indices = list(range(n))
        random.shuffle(shuffled_indices)
        for idx in shuffled_indices:
            x[idx] = 1
            if is_feasible(f, x):
                break
            x[idx] = 0
        else:
            # No feasible single-bit assignment found
            continue

        # Step 2: Greedy extension using only a sampled set of 0-bits
        zero_indices = set(i for i in range(n) if x[i] == 0)
        while True:
            if not zero_indices:
                break

            sampled_indices = random.sample(zero_indices, min(candidate_sample_size, len(zero_indices)))
            candidates = []

            for i in sampled_indices:
                x_try = x.copy()
                x_try[i] = 1
                if is_feasible(f, x_try):
                    candidates.append((f(x_try), i))

            if not candidates:
                break

            # Pick randomly among top-k
            candidates.sort(reverse=True)
            _, best_i = random.choice(candidates[:min(top_k, len(candidates))])
            x[best_i] = 1
            zero_indices.remove(best_i)

        population.append(x)

    return population

def crossover_population_position_based(f, parents, top_k_parents=20, child_pool_size=100):
    """
    Selects top `top_k_parents` individuals and performs all C(20, 2) pairwise position-based crossovers.
    Repairs all children and returns top `child_pool_size` feasible children.
    """
    # Step 1: Evaluate and select top 20 feasible parents
    scored = [(f(p), p) for p in parents if is_feasible(f, p)]
    scored.sort(reverse=True, key=lambda t: t[0])
    top_parents = [x for _, x in scored[:top_k_parents]]

    # Step 2: All 20C2 pairwise crossovers
    children = []
    for i in range(len(top_parents)):
        for j in range(i + 1, len(top_parents)):
            child = crossover_position_based(top_parents[i], top_parents[j])
            repaired = repair_solution_smart(f, child)
            children.append(repaired)

    # Step 3: Keep only top child_pool_size children based on objective value
    scored_children = [(f(c), c) for c in children if is_feasible(f, c)]
    scored_children.sort(reverse=True, key=lambda t: t[0])
    top_children = [x for _, x in scored_children[:child_pool_size]]

    return top_children





def crossover_position_based(parent1, parent2):
    """
    Perform position-based crossover between two binary parents.
    Random subset of positions are inherited from parent1, rest from parent2.
    """
    length = len(parent1)
    child = np.full(length, -1, dtype=int)  
    num_positions = random.randint(1, length // 2) 
    selected_positions = random.sample(range(length), num_positions)

    for pos in selected_positions:
        child[pos] = parent1[pos]

    for i in range(length):
        if child[i] == -1:
            child[i] = parent2[i]

    return child


# def crossover_population_position_based(parents):
#     """
#     Perform pairwise position-based crossover over parent pool.
#     Returns all children generated.
#     """
#     children = []
#     for i in range(len(parents)):
#         for j in range(i+1, len(parents)):
#             child = crossover_position_based(parents[i], parents[j])
#             children.append(child)
#     return children

def repair_solution_smart(f, x):
    """
    Repairs an infeasible solution by greedily flipping 1s to 0s.
    Prefers bit flips that make the solution feasible.
    If multiple such flips exist, chooses the one with the best objective value.
    Otherwise, chooses the bit that most reduces constraint violation.
    """
    if is_feasible(f, x):
        return x  
#     print("Repairing infeasibility")
    x = x.copy()
    current_violation = compute_violation_total(f, x)
    one_indices = np.where(x == 1)[0]

    while True:
        best_obj = float('-inf')
        best_violation = float('inf')
        best_idx = -1
        best_x = None
        found_feasible_flip = False

        for i in one_indices:
            x_try = x.copy()
            x_try[i] = 0
            violation = compute_violation_total(f, x_try)

            if violation == 0:  
                obj = f(x_try)
                if obj > best_obj:
                    best_obj = obj
                    best_idx = i
                    best_x = x_try
                    found_feasible_flip = True

            elif not found_feasible_flip:  
                reduction = current_violation - violation
                if reduction > 0 and violation < best_violation:
                    best_violation = violation
                    best_idx = i
                    best_x = x_try

        if best_x is not None:
            x = best_x
            one_indices = np.where(x == 1)[0]
            if is_feasible(f, x):
                return x
        else:
            return x

        
def mutate_feasible_bitwise(f, population, mutation_rate=0.1, fraction=0.2):
    """
    Mutate a fraction of the population (e.g., 20%) using bitwise flips.
    Each bit is flipped with `mutation_rate` only if the resulting solution is feasible.
    """
    num_to_mutate = int(len(population) * fraction)
    indices = random.sample(range(len(population)), num_to_mutate)
    mutated = []

    for idx in indices:
        x = population[idx].copy()
        for i in range(len(x)):
            if random.random() < mutation_rate:
#                 print("Mutated")
                x[i] = 1 - x[i]
                if not is_feasible(f, x):
#                     print("Reverted")
                    x[i] = 1 - x[i]  # revert
                else:
                    pass
        mutated.append(x)


    survivors = [population[i] for i in range(len(population)) if i not in indices]
    return survivors + mutated

def aggressive_mutate_population(f, population, mutation_rate=0.5):
    """
    Aggressively mutates a given population and repairs any infeasible offspring.
    """
    mutated = []

    for individual in population:
        x = individual.copy()
        for i in range(len(x)):
            if random.random() < mutation_rate:
                x[i] ^= 1  # Bit flip

        if not is_feasible(f, x):
            x = repair_solution_smart(f, x)

        if is_feasible(f, x):
            mutated.append((evaluate(f,x), x))

    mutated.sort(reverse=True, key=lambda t: t[0])
    return [x for _, x in mutated]



def quantile_based_initializations(f, pop_size_per_quantile=50, top_overall=100):
    """
    Generate solutions with fixed % of 1s (10%, 20%, ..., 90%).
    Collect all feasible ones and select top ones based on objective value.
    """
    n = f.meta_data.n_variables
    all_feasible = []
    
    
    for pct in np.arange(0.1, 10.0, 0.1):  # 0.1%, 0.2%, ..., 9.9%
        num_ones = max(1, int((pct / 100) * n))  # at least one bit set
        print(f"Generating for {pct:.1f}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))
    for pct in range(10, 100, 1):  # 10%, 20%, ..., 90%
        num_ones = int((pct / 100) * n)
        print(f"Generating for {pct}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))

    # Select top overall solutions
    all_feasible.sort(reverse=True, key=lambda t: t[0])
    top_selected = [x for _, x in all_feasible[:top_overall]]

    print(f"Generated {len(all_feasible)} feasible solutions across quantiles.")
    return top_selected





def quantile_based_initializations_2100(f, pop_size_per_quantile=50, top_overall=100):
    """
    Generate solutions with fixed % of 1s (10%, 20%, ..., 90%).
    Collect all feasible ones and select top ones based on objective value.
    """
    n = f.meta_data.n_variables
    all_feasible = []
    
    
    # for pct in np.arange(0.1, 10.0, 0.1):  # 0.1%, 0.2%, ..., 9.9%
    #     num_ones = max(1, int((pct / 100) * n))  # at least one bit set
    #     print(f"Generating for {pct:.1f}% ones --> {num_ones} bits")

    #     for _ in range(pop_size_per_quantile):
    #         x = np.zeros(n, dtype=int)
    #         ones_indices = random.sample(range(n), num_ones)
    #         x[ones_indices] = 1

    #         if is_feasible(f, x):
    #             obj = evaluate(f, x)
    #             all_feasible.append((obj, x.copy()))
    for pct in range(1, 100, 1):  # 10%, 20%, ..., 90%
        num_ones = int((pct / 100) * n)
        print(f"Generating for {pct}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))

    # Select top overall solutions
    all_feasible.sort(reverse=True, key=lambda t: t[0])
    top_selected = [x for _, x in all_feasible[:top_overall]]

    print(f"Generated {len(all_feasible)} feasible solutions across quantiles.")
    return top_selected




def quantile_based_initializations_2200(f, pop_size_per_quantile=50, top_overall=100):
    """
    Generate solutions with fixed % of 1s (10%, 20%, ..., 90%).
    Collect all feasible ones and select top ones based on objective value.
    """
    n = f.meta_data.n_variables
    all_feasible = []
    # pop_size_per_quantile = 200
    
    
    for pct in np.arange(0.01, 3.0, 0.01):  # 0.1%, 0.2%, ..., 9.9%
        num_ones = max(1, int((pct / 100) * n))  # at least one bit set
        print(f"Generating for {pct:.1f}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))
    for pct in np.arange(3, 10.0, 0.1):  # 0.1%, 0.2%, ..., 9.9%
        num_ones = max(1, int((pct / 100) * n))  # at least one bit set
        print(f"Generating for {pct:.1f}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))
    for pct in range(10, 100, 10):  # 10%, 20%, ..., 90%
        num_ones = int((pct / 100) * n)
        print(f"Generating for {pct}% ones --> {num_ones} bits")

        for _ in range(pop_size_per_quantile):
            x = np.zeros(n, dtype=int)
            ones_indices = random.sample(range(n), num_ones)
            x[ones_indices] = 1

            if is_feasible(f, x):
                obj = evaluate(f, x)
                all_feasible.append((obj, x.copy()))

    # Select top overall solutions
    all_feasible.sort(reverse=True, key=lambda t: t[0])
    top_selected = [x for _, x in all_feasible[:top_overall]]

    print(f"Generated {len(all_feasible)} feasible solutions across quantiles.")
    return top_selected



# def hybrid_initialization(f, pop_size=100, top_k=5, pop_size_per_quantile=200):
#     """
#     Combine greedy and quantile-based initializations.
#     Return top `pop_size` feasible individuals from both methods.
#     """
#     print("Generating Greedy Initialization...")
#     pop1 = init_population_greedy_ranked_extensions(f, pop_size=pop_size, top_k=top_k)

#     print("Generating Quantile-based Initialization...")
#     pop2 = quantile_based_initializations(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)

#     # Combine and keep best pop_size unique individuals
#     all_candidates = pop1 + pop2
#     scored = [(evaluate(f, x), x) for x in all_candidates if is_feasible(f, x)]

#     # Sort by objective value (descending)
#     scored.sort(reverse=True, key=lambda t: t[0])

#     top_population = [x for _, x in scored[:pop_size]]

#     print(f"Hybrid Initialization Complete: {len(top_population)} selected.")
#     return top_population



def evolutionary_search(f, problem_id, generations=100, pop_size=100, mutation_rate=0.1):
    try:
        best_solution = None
        best_objective = float('-inf')
        print("\n" + "="*40)
        print("🚀 Initializing population...")
        print("="*40)
        pop_size_per_quantile = 10
        if problem_id in range(2100,2128):
            pop_size = 10
            pop_size_per_quantile = 10
            pop1 = init_population_greedy_ranked_extensions_sampled(f, pop_size=3, top_k=5)
            pop2 = quantile_based_initializations_2100(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)
            all_candidates = pop1 + pop2
            scored = [(evaluate(f, x), x) for x in all_candidates if is_feasible(f, x)]
            scored.sort(reverse=True, key=lambda t: t[0])
            population = [x for _, x in scored[:pop_size]]
        elif problem_id in range(2200, 2225):
    #         population = smart_initialize(f, pop_size, max_ones=10)
    #         init_population_fast(f, pop_size)
    #         population = init_fast_population_greedy_ranked_extensions(f, pop_size=pop_size, top_k=5)
            pop_size_per_quantile = 20
            pop_size = 10
            population = quantile_based_initializations_2200(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)
        else:
            pop_size = 10
            pop_size_per_quantile = 10
            population = quantile_based_initializations(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)



        from collections import deque, Counter

        # Track last 10 generations' objective values
        last_objectives = deque(maxlen=10)
        stagnation_threshold = int(0.7 * pop_size)

        for gen in range(generations):
            print("\n" + "="*40)
            print(f"🌀 Generation {gen}")
            print("="*40)

            print("🔀 Performing crossover...")
            children = crossover_population_position_based(f, population, top_k_parents=20, child_pool_size=pop_size)

            print("🛠️  Repairing children...")
            children = [repair_solution_smart(f, c) for c in children]

            print("🧬 Mutating children...")
            children = mutate_feasible_bitwise(f, children, mutation_rate=mutation_rate, fraction=0.2)

            print("📊 Evaluating all candidates...")
            all_candidates = population + children
            evaluated = [(evaluate(f, x), x) for x in all_candidates if is_feasible(f, x)]
            evaluated.sort(reverse=True, key=lambda t: t[0])

            # print("🏆 Top objective values this generation:")
            # for obj, _ in evaluated[:pop_size]:
            #     print(f"   ➤ {obj:.5f}")

            if evaluated:
                if evaluated[0][0] > best_objective:
                    best_objective = evaluated[0][0]
                    best_solution = evaluated[0][1].copy()

            population = [x for _, x in evaluated[:pop_size]]

            while len(population) < pop_size:
                x = np.random.randint(0, 2, size=f.meta_data.n_variables)
                if is_feasible(f, x):
                    population.append(x)
            # Track and detect stagnation
            current_objs = [obj for obj, _ in evaluated[:pop_size]]
            last_objectives.append(current_objs)

            if len(last_objectives) == 10:
                flat_objectives = [obj for gen_objs in last_objectives for obj in gen_objs]
                most_common = Counter(flat_objectives).most_common(1)

                if most_common and most_common[0][1] >= stagnation_threshold * 10:
                    print("\n⚠️  Detected stagnation: Reinitializing with hybrid strategy...\n")
                    f.reset()

                    # Half from smart initialization
                    if problem_id in range(2100, 2128):
                        pop1 = init_population_greedy_ranked_extensions_sampled(f, pop_size=3, top_k=5)
                        pop2 = quantile_based_initializations(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)
                        all_candidates = pop1 + pop2
                    elif problem_id in range(2200, 2225):
                        population = quantile_based_initializations_2200(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)
                    else:
                        all_candidates = quantile_based_initializations(f, pop_size_per_quantile=pop_size_per_quantile, top_overall=pop_size)
                    print("length of all candidates = ", len(all_candidates))
                    scored_init = [(evaluate(f, x), x) for x in all_candidates if is_feasible(f, x)]
                    scored_init.sort(reverse=True, key=lambda t: t[0])
                    best_from_init = [x for _, x in scored_init[:pop_size]]

                    # Half from aggressive mutation of current population
                    # Try multiple mutation rates
                    # mutation_rates = [0.5, 0.6]
                    # all_mutated = []

                    # for rate in mutation_rates:
                    #     mutated = aggressive_mutate_population(f, population, mutation_rate=rate)
                    #     all_mutated.extend(mutated)

                    # # Evaluate and sort
                    # scored_mutated = [(evaluate(f, x), x) for x in all_mutated if is_feasible(f, x)]
                    # scored_mutated.sort(reverse=True, key=lambda t: t[0])

                    # # Select top half of the population from mutated candidates
                    # best_from_mutation = [x for _, x in scored_mutated[:pop_size // 2]]

                    # Combine and reset
                    population = best_from_init #+ best_from_mutation
                    last_objectives.clear()
                    continue


            print(f"✅ Best so far: {best_objective:.5f}")
            print("-" * 40)
    except:
        print("BUDGET EXCEEDED!")
        print(f"Best feasible objective value: {best_objective}")
        print(f"Best solution: {best_solution}")
        return best_solution
    
    print(f"Best feasible objective value: {best_objective}")
    print(f"Best solution: {best_solution}")
    return best_solution




if __name__ == "__main__":
    print("Welcome to the Constraint-Aware Evolutionary Search!")
    parser = argparse.ArgumentParser(description="Run Evolutionary Search with IOHProfiler")
    parser.add_argument("--problem_id", type=int, default=2100, help="ID of the IOH problem (default: 2100)")
    parser.add_argument("--budget", type=int, default=10000, help="Budget for evaluations (default: 10000)")
    generations = 1000
    pop_size = 30
    mutation_rate = 0.1

    args = parser.parse_args()
    budget = args.budget

    logger = ioh.logger.Analyzer(
        root=os.getcwd(),
        folder_name="constraint-aware-ea",
        algorithm_name="evolutionary-search",
        store_positions=False
    )

    f = ioh.get_problem(args.problem_id, problem_class=ioh.ProblemClass.GRAPH)
    f.attach_logger(logger)

    solution = evolutionary_search(f, args.problem_id, generations=generations,
                                   pop_size=pop_size, mutation_rate=mutation_rate)
    
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("🚀 Evolutionary Search Complete! Solution is : ")
    if solution is not None:
        if f(solution) >f.state.current_best.y:
            print(solution)
        else:
            print(f.state.current_best)
            print(f.state)
            print(f"Best objective value found: {f.state.current_best.y:.5f}")
            print("Evaluation count:", f.state.evaluations)
            print("Our Evaluation count:", eval_count)
    else:
        print(f.state.current_best)
        print(f.state)
        print(f"Best objective value found: {f.state.current_best.y:.5f}")
        print("Evaluation count:", f.state.evaluations)
        print("Our Evaluation count:", eval_count)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    print("\n" + "="*40)
    f.reset()
    logger.close()





