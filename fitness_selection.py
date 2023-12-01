import random
import bisect

def fitness_select(population, elites=4):
    fitness_values = [individual.fitness for individual in population]
    probability_intervals = [sum(fitness_values[:i + 1]) for i in range(len(fitness_values))]

    def select_individual():
        random_select = random.uniform(0, probability_intervals[-1])
        selected_index = bisect.bisect_left(probability_intervals, random_select)
        return population[selected_index]

    selected_pairs = []
    for i in range(len(population) - elites):
        first_individual, second_individual = select_individual(), select_individual()
        selected_pairs.append((first_individual, second_individual))

    return selected_pairs