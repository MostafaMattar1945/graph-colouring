import random

#1.Create one chromosome(solution)
def create_individual(n_genes, k, rng=random):
    return [rng.randrange(k) for _ in range(n_genes)]

#---------------------------------------------------------------------------------------------------
def create_population(pop_size, n_genes, k, rng=random):
    
    population = []  # Create an empty list to store the chromosomes

    for _ in range(pop_size):  # Repeat the process to create the required number of chromosomes
        individual = create_individual(n_genes, k, rng)  # Create one random chromosome
        population.append(individual)  # Add the chromosome to the population

    return population  # Return the complete population
#---------------------------------------------------------------------------------------------------

def conflicts_of(individual, g):
    """
    Calculate the number of conflicts in a chromosome.

    individual: A chromosome represented as a list of integer colours.
    g: The graph being solved.

    Returns:
        The number of conflicts in the chromosome.
    """

    # Convert the chromosome into a dictionary:
    # Each vertex is mapped to its corresponding colour.
    colouring = {}

    for i in range(len(g.vertices)):
        vertex = g.vertices[i]
        colour = individual[i]

        colouring[vertex] = colour

    # Use the Graph's existing method to count conflicts.
    conflicts = g.count_conflicts(colouring)

    return conflicts
#---------------------------------------------------------------------------------------------------


def fitness(individual, g):
    """
    Calculate the fitness score of a chromosome.
    Returns:
        A fitness score between 0 and 1.
        Higher scores represent better solutions.
    """

    # Calculate the number of conflicts in this chromosome.
    conflicts = conflicts_of(individual, g)

    # Convert the number of conflicts into a fitness score.
    score = 1 / (1 + conflicts)

    return score

#---------------------------------------------------------------------------------------------------
def tournament_selection(population, fitness_fn, num_parents,
                         tournament_size=3, rng=random):
    """
    Select parents using Tournament Selection.
    tournament_size: Number of individuals competing in each tournament.
    rng: Random number generator.

    Returns:
        A list containing the selected parent chromosomes.
    """

    parents = []  # Store the selected parents

    for _ in range(num_parents):

        # Randomly select individuals to compete in the tournament.
        contestants = rng.sample(population, tournament_size)

        # Assume the first contestant is the winner initially.
        winner = contestants[0]

        # Compare the fitness of the remaining contestants.
        for individual in contestants[1:]:

            if fitness_fn(individual) > fitness_fn(winner):
                winner = individual

        # Add the winner to the parents list.
        parents.append(winner)

    return parents
#---------------------------------------------------------------------------------------------------
def roulette_wheel_selection(population, fitness_fn, num_parents, rng=random):

    # Calculate fitness for each individual.
    fitness_scores = []

    for individual in population:
        score = fitness_fn(individual)
        fitness_scores.append(score)

    # Calculate total fitness.
    total_fitness = sum(fitness_scores)

    # If total fitness is zero, select parents randomly.
    if total_fitness == 0:
        parents = []

        for _ in range(num_parents):
            selected_parent = rng.choice(population)
            parents.append(selected_parent)

        return parents

    # Calculate selection probabilities.
    probabilities = []

    for score in fitness_scores:
        probability = score / total_fitness
        probabilities.append(probability)

    # Select parents using roulette wheel selection.
    parents = []

    for _ in range(num_parents):
        selected_index = rng.choices(
            range(len(population)),
            weights=probabilities,
            k=1
        )[0]

        parents.append(population[selected_index])

    return parents
#---------------------------------------------------------------------------------------------------
def one_point_crossover(parent1, parent2, rng=random):

    # Both parents must have the same length.
    if len(parent1) != len(parent2):
        raise ValueError("Both parents must have the same length.")

    # If chromosomes have fewer than 2 genes, return copies.
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    # Choose a random crossover point.
    crossover_point = rng.randint(1, len(parent1) - 1)

    # Create two children by exchanging parts of the parents.
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]

    return child1, child2

#---------------------------------------------------------------------------------------------------
def mutate(individual, k, mutation_rate, rng=random):

    # Create a copy of the original individual.
    mutated_individual = individual.copy()

    # If there is only one available colour, return without mutation.
    if k == 1:
        return mutated_individual

    # Check each gene for mutation.
    for i in range(len(mutated_individual)):

        if rng.random() < mutation_rate:

            available_colours = []

            for colour in range(k):
                if colour != mutated_individual[i]:
                    available_colours.append(colour)

            new_colour = rng.choice(available_colours)
            mutated_individual[i] = new_colour

    return mutated_individual