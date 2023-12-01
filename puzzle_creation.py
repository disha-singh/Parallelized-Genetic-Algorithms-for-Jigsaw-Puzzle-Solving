import cv2
import numpy as np

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