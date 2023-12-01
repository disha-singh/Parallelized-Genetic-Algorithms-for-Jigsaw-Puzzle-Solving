import cv2
import numpy as np
from operator import attrgetter
from image_analysis import divide_image_into_pieces, ImageAnalysis
from crossover_operator import CrossoverOperator
from individual import Individual
from puzzle_creation import fitness_select

class GeneticAlgorithm:
    TERMINATION_THRESHOLD = 100

    def __init__(self, original_image, piece_size, population_size, generations, elite_size=2):
        if original_image is None:
            raise ValueError("Error: Unable to read the original image.")

        self._original_image = original_image
        self._piece_size = piece_size
        self._generations = generations
        self._elite_size = elite_size

        puzzle_pieces, rows, columns = divide_image_into_pieces(original_image, piece_size, indexed=True)
        self._population = [Individual(puzzle_pieces, rows, columns) for _ in range(population_size)]
        self._puzzle_pieces = puzzle_pieces

    def start_evolution(self):
        print("Puzzle Pieces:\t{}\n".format(len(self._puzzle_pieces)))

        ImageAnalysis.analyze_puzzle_pieces(self._puzzle_pieces)

        best_individual = None  # Best individual of a generation
        best_fitness_score = float("-inf")

        for generation in range(self._generations):

            new_population = []

            elite_individuals = self._get_best_individuals(elites=self._elite_size)
            new_population.extend(elite_individuals)

            selected_parents = fitness_select(self._population, elites=self._elite_size)

            for first_parent, second_parent in selected_parents:
                crossover_operator = CrossoverOperator(first_parent, second_parent)
                crossover_operator.execute_crossover()
                child_individual = crossover_operator.create_child()
                new_population.append(child_individual)

            best_individual = self._best_individual()

            if best_individual.fitness > best_fitness_score:
                best_fitness_score = best_individual.fitness

            self._population = new_population

            # This saves the best individual of every generation. Change the path here.
            generation_str = str(generation + 1)
            image_output_path = "/Images/output_solution/gen" + generation_str + ".jpg"
            cv2.imwrite(image_output_path, best_individual.to_image())

        return best_individual

    def _get_best_individuals(self, elites):
        return sorted(self._population, key=attrgetter("fitness"))[-elites:]

    def _best_individual(self):
        return max(self._population, key=attrgetter("fitness"))