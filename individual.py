import random
import numpy as np
import cv2
from image_analysis import ImageAnalysis
from puzzle_creation import PuzzlePiece
from puzzle_creation import assemble_image_from_pieces

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
