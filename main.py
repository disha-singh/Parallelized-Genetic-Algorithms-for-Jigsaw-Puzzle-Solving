import cv2
import time
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib import gridspec
from puzzle_creation import create_puzzle, PuzzlePiece, divide_image_into_pieces, assemble_image_from_pieces
from crossover_operator import CrossoverOperator, complementary_orientation
from individual import Individual, fitness_select
from genetic_algorithm import GeneticAlgorithm, ParallelGeneticAlgorithm
from image_analysis import ImageAnalysis

# Main Program
start_time_puzzle_creation = time.perf_counter()

# CHANGE PIECE SIZE HERE (bigger value = fewer pieces)
piece_size_for_puzzle_creation = 64

# CHANGE IMAGE INPUT / OUTPUT HERE
original_image_input = "/Images/Image3.jpg"
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