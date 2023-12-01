import cv2
import numpy as np

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
