import numpy as np
import cv2
import time
from operator import attrgetter
import heapq
import random
import bisect
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib import gridspec

SHARED_PRIORITY = -10
BUDDY_PRIORITY = -1

# Create Puzzle
class PuzzlePiece:
    def __init__(self, image_data, piece_id):
        self.image_data = image_data[:]
        self.piece_id = piece_id

    def __getitem__(self, index):
        return self.image_data.__getitem__(index)

    def size(self):
        return self.image_data.shape[0]

    def shape(self):
        return self.image_data.shape


def divide_image_into_pieces(original_image, piece_size, indexed=False):
    rows, columns = original_image.shape[0] // piece_size, original_image.shape[1] // piece_size
    pieces = []

    for y in range(rows):
        for x in range(columns):
            left, top, right, bottom = x * piece_size, y * piece_size, (x + 1) * piece_size, (y + 1) * piece_size
            piece_data = np.empty((piece_size, piece_size, original_image.shape[2]))
            piece_data[:piece_size, :piece_size, :] = original_image[top:bottom, left:right, :]
            pieces.append(piece_data)

    if indexed:
        pieces = [PuzzlePiece(piece_value, index) for index, piece_value in enumerate(pieces)]

    return pieces, rows, columns


def assemble_image_from_pieces(puzzle_pieces, rows, columns):
    vertical_stack = []
    for i in range(rows):
        horizontal_stack = []
        for j in range(columns):
            horizontal_stack.append(puzzle_pieces[i * columns + j])
        vertical_stack.append(np.hstack(horizontal_stack))
    return np.vstack(vertical_stack).astype(np.uint8)


def create_puzzle(input_image_path, output_image_path, piece_size):
    original_image = cv2.imread(input_image_path)
    puzzle_pieces, rows, columns = divide_image_into_pieces(original_image, piece_size)

    np.random.shuffle(puzzle_pieces)

    puzzled_image = assemble_image_from_pieces(puzzle_pieces, rows, columns)

    cv2.imwrite(output_image_path, puzzled_image)
    print("Puzzle created with {} pieces\n".format(len(puzzle_pieces)))


class CrossoverOperator:
    def __init__(self, parent1, parent2):
        self._parents = (parent1, parent2)
        self._piece_count = len(parent1.puzzle_pieces)
        self._child_rows = parent1.rows
        self._child_columns = parent1.columns

        self._min_row = 0
        self._max_row = 0
        self._min_column = 0
        self._max_column = 0

        self._main_pieces = {}
        self._taken_positions = set()

        self._candidate_pieces = []

    def create_child(self):
        child_pieces = [None] * self._piece_count

        for piece_id, (row, column) in self._main_pieces.items():
            index = (row - self._min_row) * self._child_columns + (column - self._min_column)
            child_pieces[index] = self._parents[0].piece_by_id(piece_id)

        return Individual(child_pieces, self._child_rows, self._child_columns, shuffle=False)

    def execute_crossover(self):
        self._initialize_main()

        while len(self._candidate_pieces) > 0:
            _, (position, piece_id), relative_piece = heapq.heappop(self._candidate_pieces)

            if position in self._taken_positions:
                continue

            if piece_id in self._main_pieces:
                self.add_piece_to_possible(position, relative_piece[0], relative_piece[1])
                continue

            self._add_piece_to_main(piece_id, position)

    def _initialize_main(self):
        root_piece = self._parents[0].get_random_piece()
        self._add_piece_to_main(root_piece.piece_id, (0, 0))

    def _add_piece_to_main(self, piece_id, position):
        self._main_pieces[piece_id] = position
        self._taken_positions.add(position)
        self._update_possible_pieces(piece_id, position)

    def _update_possible_pieces(self, piece_id, position):
        available_boundaries = self._available_edges(position)

        for orientation, position in available_boundaries:
            self.add_piece_to_possible(position, piece_id, orientation)

    def add_piece_to_possible(self, position, piece_id, orientation):
        shared_piece = self._get_shared_piece(piece_id, orientation)

        if self._is_valid_piece(shared_piece):
            self._add_shared_piece_possible(shared_piece, position, (piece_id, orientation))
            return

        buddy_piece = self._get_buddy_piece(piece_id, orientation)

        if self._is_valid_piece(buddy_piece):
            self._add_buddy_piece_possible(buddy_piece, position, (piece_id, orientation))
            return

        best_match_piece, priority = self._get_best_match_piece(piece_id, orientation)

        if self._is_valid_piece(best_match_piece):
            self._add_best_match_piece_possible(best_match_piece, position, priority, (piece_id, orientation))
            return

    def _get_shared_piece(self, piece_id, orientation):
        first_parent, second_parent = self._parents
        first_parent_edge = first_parent.get_edge(piece_id, orientation)
        second_parent_edge = second_parent.get_edge(piece_id, orientation)

        if first_parent_edge == second_parent_edge:
            return first_parent_edge

    def _get_buddy_piece(self, piece_id, orientation):
        first_buddy = ImageAnalysis.best_match(piece_id, orientation)
        second_buddy = ImageAnalysis.best_match(first_buddy, complementary_orientation(orientation))

        if second_buddy == piece_id:
            for edge in [parent.get_edge(piece_id, orientation) for parent in self._parents]:
                if edge == first_buddy:
                    return edge

    def _get_best_match_piece(self, piece_id, orientation):
        for piece, dissimilarity_measure in ImageAnalysis.piece_best_match_table[piece_id][orientation]:
            if self._is_valid_piece(piece):
                return piece, dissimilarity_measure

    def _add_shared_piece_possible(self, piece_id, position, relative_piece):
        piece_candidate = (SHARED_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_buddy_piece_possible(self, piece_id, position, relative_piece):
        piece_candidate = (BUDDY_PRIORITY, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _add_best_match_piece_possible(self, piece_id, position, priority, relative_piece):
        piece_candidate = (priority, (position, piece_id), relative_piece)
        heapq.heappush(self._candidate_pieces, piece_candidate)

    def _available_edges(self, row_and_column):
        (row, column) = row_and_column
        edges = []

        if not self._main_full():
            positions = {
                "T": (row - 1, column),
                "R": (row, column + 1),
                "D": (row + 1, column),
                "L": (row, column - 1)
            }

            for orientation, position in positions.items():
                if position not in self._taken_positions and self._in_range(position):
                    self._update_main_edges(position)
                    edges.append((orientation, position))

        return edges

    def _main_full(self):
        return len(self._main_pieces) == self._piece_count

    def _in_range(self, row_and_column):
        (row, column) = row_and_column
        return self._row_in_range(row) and self._column_in_range(column)

    def _row_in_range(self, row):
        current_rows = abs(min(self._min_row, row)) + abs(max(self._max_row, row))
        return current_rows < self._child_rows

    def _column_in_range(self, column):
        current_columns = abs(min(self._min_column, column)) + abs(max(self._max_column, column))
        return current_columns < self._child_columns

    def _update_main_edges(self, row_and_column):
        (row, column) = row_and_column
        self._min_row = min(self._min_row, row)
        self._max_row = max(self._max_row, row)
        self._min_column = min(self._min_column, column)
        self._max_column = max(self._max_column, column)

    def _is_valid_piece(self, piece_id):
        return piece_id is not None and piece_id not in self._main_pieces


def complementary_orientation(orientation):
    return {
        "T": "D",
        "R": "L",
        "D": "T",
        "L": "R"
    }.get(orientation, None)


class Individual:
    FITNESS_FACTOR = 1000

    def __init__(self, puzzle_pieces, rows, columns, shuffle=True):
        self.puzzle_pieces = puzzle_pieces[:]
        self.rows = rows
        self.columns = columns
        self._fitness = None

        if shuffle:
            np.random.shuffle(self.puzzle_pieces)

        self._piece_mapping = {piece.piece_id: index for index, piece in enumerate(self.puzzle_pieces)}

    def __getitem__(self, key):
        return self.puzzle_pieces[key * self.columns:(key + 1) * self.columns]

    @property
    def fitness(self):
        if self._fitness is None:
            fitness_value = 1 / self.FITNESS_FACTOR

            for i in range(self.rows):
                for j in range(self.columns - 1):
                    ids = (self[i][j].piece_id, self[i][j + 1].piece_id)
                    fitness_value += ImageAnalysis.calculate_difference_measure(self[i][j], self[i][j + 1], orientation="LR")

            for i in range(self.rows - 1):
                for j in range(self.columns):
                    ids = (self[i][j].piece_id, self[i + 1][j].piece_id)
                    fitness_value += ImageAnalysis.calculate_difference_measure(self[i][j], self[i + 1][j], orientation="TD")

            self._fitness = self.FITNESS_FACTOR / fitness_value

        return self._fitness

    def piece_size(self):
        return self.puzzle_pieces[0].size()

    def piece_by_id(self, identifier):
        return self.puzzle_pieces[self._piece_mapping[identifier]]

    def to_image(self):
        pieces = [piece.image_data for piece in self.puzzle_pieces]
        return assemble_image_from_pieces(pieces, self.rows, self.columns)

    def get_edge(self, piece_id, orientation):
        edge_index = self._piece_mapping[piece_id]

        if (orientation == "T") and (edge_index >= self.columns):
            return self.puzzle_pieces[edge_index - self.columns].piece_id

        if (orientation == "R") and (edge_index % self.columns < self.columns - 1):
            return self.puzzle_pieces[edge_index + 1].piece_id

        if (orientation == "D") and (edge_index < (self.rows - 1) * self.columns):
            return self.puzzle_pieces[edge_index + self.columns].piece_id

        if (orientation == "L") and (edge_index % self.columns > 0):
            return self.puzzle_pieces[edge_index - 1].piece_id

    def get_random_piece(self):
        return random.choice(self.puzzle_pieces)


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

class ParallelGeneticAlgorithm(GeneticAlgorithm):
    def start_evolution(self):
        print("Puzzle Pieces:\t{}\n".format(len(self._puzzle_pieces)))
        ImageAnalysis.analyze_puzzle_pieces(self._puzzle_pieces)

        best_individual = None
        best_fitness_score = float("-inf")

        with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor instead of ThreadPoolExecutor
            for generation in range(self._generations):
                new_population = []

                elite_individuals = self._get_best_individuals(elites=self._elite_size)
                new_population.extend(elite_individuals)

                selected_parents = fitness_select(self._population, elites=self._elite_size)

                # Use process pool for parallelization
                crossover_jobs = [executor.submit(self._crossover_worker, first_parent, second_parent)
                                  for first_parent, second_parent in selected_parents]

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


# Image Analysis
# Image Analysis
class ImageAnalysis(object):
    piece_difference_measures = {}
    piece_best_match_table = {}

    @classmethod
    def analyze_puzzle_pieces(cls, puzzle_pieces):
        for piece in puzzle_pieces:
            cls.piece_best_match_table[piece.piece_id] = {
                "T": [],
                "R": [],
                "D": [],
                "L": []
            }

        def update_best_match_table(first_piece, second_piece, orientation):
            measure = cls.calculate_difference_measure(first_piece, second_piece, orientation)
            cls.store_difference((first_piece.piece_id, second_piece.piece_id), orientation, measure)
            cls.piece_best_match_table[second_piece.piece_id][orientation[0]].append((first_piece.piece_id, measure))
            cls.piece_best_match_table[first_piece.piece_id][orientation[1]].append((second_piece.piece_id, measure))

        iterations = len(puzzle_pieces) - 1
        for first in range(iterations):
            for second in range(first + 1, len(puzzle_pieces)):
                for orientation in ["LR", "TD"]:
                    update_best_match_table(puzzle_pieces[first], puzzle_pieces[second], orientation)
                    update_best_match_table(puzzle_pieces[second], puzzle_pieces[first], orientation)

        for piece in puzzle_pieces:
            for orientation in ["T", "L", "R", "D"]:
                cls.piece_best_match_table[piece.piece_id][orientation].sort(key=lambda x: x[1])

    @classmethod
    def store_difference(cls, ids, orientation, value):
        if ids not in cls.piece_difference_measures:
            cls.piece_difference_measures[ids] = {}
        cls.piece_difference_measures[ids][orientation] = value

    @classmethod
    def get_difference(cls, ids, orientation):
        return cls.piece_difference_measures[ids][orientation]

    @classmethod
    def best_match(cls, piece_id, orientation):
        return cls.piece_best_match_table[piece_id][orientation][0][0]

    @classmethod
    def calculate_difference_measure(cls, first_piece, second_piece, orientation="LR"):
        rows, columns, _ = first_piece.shape()
        color_difference = None

        # HORIZONTAL (LEFT - RIGHT)
        if orientation == "LR":
            color_difference = first_piece[:, columns - 1, :] - second_piece[:, 0, :]

        # VERTICAL (TOP - DOWN)
        if orientation == "TD":
            color_difference = first_piece[rows - 1, :, :] - second_piece[0, :, :]

        if color_difference is not None:
            squared_color_difference = np.power(color_difference / 255.0, 2)
            color_difference_per_row = np.sum(squared_color_difference, axis=1)
            total_difference = np.sum(color_difference_per_row, axis=0)

            value = np.sqrt(total_difference)

            return value

        return 0.0

# Main Program
start_time_puzzle_creation = time.perf_counter()

# CHANGE PIECE SIZE HERE (bigger value = fewer pieces)
piece_size_for_puzzle_creation = 64

# CHANGE IMAGE INPUT / OUTPUT HERE
original_image_input = "/Images/Image4.jpeg"
puzzled_image_output = "/Images/output_puzzle/puzzled_out.jpg"

create_puzzle(original_image_input, puzzled_image_output, piece_size_for_puzzle_creation)

finish_time_puzzle_creation = time.perf_counter()

# CHANGE generation and population number here
GENERATIONS = 30
POPULATION = 300

# Load puzzled image
puzzled_image = cv2.imread(puzzled_image_output)

# Puzzle solving with serial genetic algorithm
start_time_serial_genetic = time.perf_counter()

solution_serial = GeneticAlgorithm(puzzled_image, piece_size_for_puzzle_creation, POPULATION, GENERATIONS).start_evolution().to_image()

# Saving the serial output.
image_output_serial = "/Images/output_solution/solution_serial.jpg"
cv2.imwrite(image_output_serial, solution_serial)
print("Saved to '{}'".format(image_output_serial))

finish_time_serial_genetic = time.perf_counter()
serial_execution_time = finish_time_serial_genetic - start_time_serial_genetic

# Puzzle solving with parallel genetic algorithm
start_time_parallel_genetic = time.perf_counter()

solution_parallel = ParallelGeneticAlgorithm(puzzled_image, piece_size_for_puzzle_creation, POPULATION, GENERATIONS).start_evolution().to_image()

# Saving the parallel output.
image_output_parallel = "/Images/output_solution/solution_parallel.jpg"
cv2.imwrite(image_output_parallel, solution_parallel)
print("Saved to '{}'".format(image_output_parallel))

finish_time_parallel_genetic = time.perf_counter()
parallel_execution_time = finish_time_parallel_genetic - start_time_parallel_genetic

# Calculate speedup and efficiency
speedup = serial_execution_time / parallel_execution_time
efficiency = speedup / 4  # Assuming 4 cores, adjust as needed

# Display execution times
print("Puzzle Creation Time: {:.2f} seconds".format(finish_time_puzzle_creation - start_time_puzzle_creation))
print("Serial Genetic Algorithm Execution Time: {:.2f} seconds".format(serial_execution_time))
print("Parallel Genetic Algorithm Execution Time: {:.2f} seconds".format(parallel_execution_time))
print("Speedup: {:.2f}".format(speedup))
print("Efficiency: {:.2%}".format(efficiency))

# Resize images for display
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height))

scale_percent = 30
# Load images for display
original_image = cv2.imread(original_image_input)
puzzled_image_resized = resize_image(puzzled_image, scale_percent)
solution_serial_resized = resize_image(solution_serial, scale_percent)
solution_parallel_resized = resize_image(solution_parallel, scale_percent)

# Create subplots
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# Plot images
ax0 = plt.subplot(gs[0])
ax0.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
ax0.set_title('Original Image')
ax0.axis('off')

ax1 = plt.subplot(gs[1])
ax1.imshow(cv2.cvtColor(puzzled_image_resized, cv2.COLOR_BGR2RGB))
ax1.set_title('Puzzled Image')
ax1.axis('off')

ax2 = plt.subplot(gs[2])
ax2.imshow(cv2.cvtColor(solution_serial_resized, cv2.COLOR_BGR2RGB))
ax2.set_title('Solution (Serial)')
ax2.axis('off')

ax3 = plt.subplot(gs[3])
ax3.imshow(cv2.cvtColor(solution_parallel_resized, cv2.COLOR_BGR2RGB))
ax3.set_title('Solution (Parallel)')
ax3.axis('off')

plt.tight_layout()
plt.show()