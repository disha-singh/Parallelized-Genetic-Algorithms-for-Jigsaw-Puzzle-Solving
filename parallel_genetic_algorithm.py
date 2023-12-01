import cv2
import numpy as np
from operator import attrgetter
import concurrent.futures
from genetic_algorithm import GeneticAlgorithm
from crossover_operator import CrossoverOperator
from individual import Individual
from puzzle_creation import fitness_select
from image_analysis import divide_image_into_pieces, ImageAnalysis

class ParallelGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_evolution(self):
        print("Puzzle Pieces:\t{}\n".format(len(self._puzzle_pieces)))

        ImageAnalysis.analyze_puzzle_pieces(self._puzzle_pieces)

        best_individual = None  # Best individual of a generation
        best_fitness_score = float("-inf")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for generation in range(self._generations):
                new_population = []

                elite_individuals = self._get_best_individuals(elites=self._elite_size)
                new_population.extend(elite_individuals)

                selected_parents = fitness_select(self._population, elites=self._elite_size)

                # Parallelize the crossover process
                crossover_jobs = [executor.submit(self._crossover_worker, first_parent, second_parent) for
                                  first_parent, second_parent in selected_parents]

                # Wait for all crossover jobs to complete
                concurrent.futures.wait(crossover_jobs)

                # Retrieve the results
                new_population.extend(job.result() for job in crossover_jobs)

                best_individual = self._best_individual()

                if best_individual.fitness > best_fitness_score:
                    best_fitness_score = best_individual.fitness

                self._population = new_population

                # Save the best individual of every generation. Change the path here.
                generation_str = str(generation + 1)
                image_output_path = "/Images/output_solution/gen" + generation_str + ".jpg"
                cv2.imwrite(image_output_path, best_individual.to_image())

        return best_individual

    def _crossover_worker(self, first_parent, second_parent):
        crossover_operator = CrossoverOperator(first_parent, second_parent)
        crossover_operator.execute_crossover()
        return crossover_operator.create_child()