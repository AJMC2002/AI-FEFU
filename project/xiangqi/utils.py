import os
import sys

sys.path.append(os.path.relpath("trainer"))

from numpy import expand_dims, ndarray
from tensorflow.keras.models import Model
from train import encode_fen


def pos2index(pos: str) -> int:
    return 9 * (9 - int(pos[1])) + (ord(pos[0]) - ord("a"))


def get_probability_vectors(model: Model, fen: str) -> tuple[ndarray, ndarray]:
    board = expand_dims(encode_fen(fen), axis=0)

    from_tensor, to_tensor = model.predict(board, verbose=0)
    from_vec = from_tensor[0]
    to_vec = to_tensor[0]

    return from_vec, to_vec


def move_probability(from_vec: ndarray, to_vec: ndarray, move: str) -> float:
    from_prob = from_vec[pos2index(move[:2])]
    to_prob = to_vec[pos2index(move[2:])]

    return from_prob * to_prob
