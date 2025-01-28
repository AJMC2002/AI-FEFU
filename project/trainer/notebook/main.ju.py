# %% [md]
# # Project: Chinese Chess cDNN Model
# First install dependencies.

# %%
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.keras as tfk
from bidict import bidict
from tensorflow.keras import layers, models

tfk.mixed_precision.set_global_policy("mixed_float16")

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DATA_PATH = "../data/MOVES.DAT"
BOARD_SIZE = (10, 9)

# %% [md]
# Now we creat bidirectional dictionaries in order to encode and decode the pieces and moves.
#
# PIECE_VAL encodes the pieces to some int.
#
# MOVE_ID encodes the best move from FEN format to an int.

# %%
PIECE_VAL = bidict(
    {
        "K": 1,  # Red King
        "A": 2,  # Red Advisor
        "B": 3,  # Red Elephant
        "N": 4,  # Red Horse
        "R": 5,  # Red Chariot
        "C": 6,  # Red Cannon
        "P": 7,  # Red Pawn
        "k": 8,  # Black King
        "a": 9,  # Black Advisor
        "b": 10,  # Black Elephant
        "n": 11,  # Black Horse
        "r": 12,  # Black Chariot
        "c": 13,  # Black Cannon
        "p": 14,  # Black Pawn
    }
)

print(PIECE_VAL)

# %%
columns = "abcdefghi"
rows = range(0, 10)
moves: set[str] = set()

# unbounded moves
for r in rows:
    for c in range(len(columns)):
        from_pos = f"{columns[c]}{r}"
        # horizontal moves
        for cc in range(len(columns)):
            if cc != c:
                to_pos = f"{columns[cc]}{r}"
                moves.add(f"{from_pos}{to_pos}")
        # vertical moves
        for rr in rows:
            if rr != r:
                to_pos = f"{columns[c]}{rr}"
                moves.add(f"{from_pos}{to_pos}")
        # horse moves
        horse_directions = [
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
        ]
        from_pos = f"{columns[c]}{r}"
        for dc, dr in horse_directions:
            nc, nr = c + dc, r + dr
            if 0 <= nc < len(columns) and 0 <= nr < 10:
                to_pos = f"{columns[nc]}{nr}"
                moves.add(f"{from_pos}{to_pos}")

# bounded moves
# elephants
r_elephant = [
    ["a2", "c0"],
    ["a2", "c4"],
    ["c0", "e2"],
    ["c4", "e2"],
    ["e2", "g0"],
    ["e2", "g4"],
    ["g0", "i2"],
    ["g4", "i2"],
]
b_elephant = [
    ["a7", "c5"],
    ["a7", "c9"],
    ["c5", "e7"],
    ["c9", "e7"],
    ["e7", "g5"],
    ["e7", "g9"],
    ["g5", "i7"],
    ["g9", "i7"],
]
# avisor
r_advisor = [["e1", "d0"], ["e1", "d2"], ["e1", "f0"], ["e1", "f2"]]
b_advisor = [["e8", "d9"], ["e8", "d7"], ["e8", "f9"], ["e8", "f7"]]
# kings
r_king = [["e1", "e0"], ["e1", "e2"], ["e1", "d1"], ["e1", "f1"]]
b_king = [["e8", "e9"], ["e8", "e7"], ["e8", "d8"], ["e8", "f8"]]
# adding to moves
for from_pos, to_pos in [
    *r_elephant,
    *b_elephant,
    *r_advisor,
    *b_advisor,
    *r_king,
    *b_king,
]:
    moves.add(f"{from_pos}{to_pos}")
    moves.add(f"{to_pos}{from_pos}")

MOVE_ID = bidict({move: idx for idx, move in enumerate(moves)})

print(MOVE_ID)


# %% [md]
# `encode_fen` takes a fen string representing the board position, and will make it into an array representing pieces using `PIECE_VAL`. `decode_fen` does the oppositte.


# %%
def encode_fen(fen: str) -> npt.NDArray[np.int_]:
    board = np.zeros((10, 9), dtype=int)
    rows = fen.split()[0].split("/")
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col] = PIECE_VAL[char]
                col += 1
    return board


def decode_fen(fen_board: npt.NDArray[np.int_], turn: str) -> str:
    fen_rows = []
    for row in fen_board:
        fen_row = ""
        empty_count = 0
        for piece in row:
            if piece == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += PIECE_VAL.inv[piece]
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)
    turn = "w" if turn[0].lower() == "r" or turn[0].lower() == "w" else "b"
    return f"{fen} {turn}"


# Test
test_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
test_board = np.array(
    [
        [12, 11, 10, 9, 8, 9, 10, 11, 12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 13, 0, 0, 0, 0, 0, 13, 0],
        [14, 0, 14, 0, 14, 0, 14, 0, 14],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 0, 7, 0, 7, 0, 7, 0, 7],
        [0, 6, 0, 0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5, 4, 3, 2, 1, 2, 3, 4, 5],
    ]
)
assert (encode_fen(test_fen) == test_board).all()
assert decode_fen(test_board, "red") == test_fen


# %% [md]
# `encode_move` uses the previously generated dictionary to encode the moves from the training data. `decode_move` does the oppositte.


# %%
def encode_move(move: str) -> npt.NDArray[np.int_]:
    move_id = MOVE_ID[move]
    one_hot_move = np.zeros(len(MOVE_ID), dtype=np.int_)
    one_hot_move[move_id] = 1
    return one_hot_move


def decode_move(one_hot_move: npt.NDArray[np.float_]) -> str:
    move_id = int(np.argmax(one_hot_move))
    return MOVE_ID.inv[move_id]


# Test
test_move_id = np.argmax(encode_move("b2e2"))
test_one_hot_move = np.zeros(len(MOVE_ID), dtype=np.float_)
test_one_hot_move[test_move_id] = 1
assert (encode_move("b2e2") == test_one_hot_move).all()
assert decode_move(test_one_hot_move) == "b2e2"


# %% [md]
# `load_data` will read a file containing the best move of a position, the weight of the move, and the FEN string representing the current position.


# %%
def load_data(file_path: str) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    fens: list[npt.NDArray[np.int_]] = []
    best_moves: list[npt.NDArray[np.int_]] = []
    with open(file_path, "r") as f:
        for line in f:
            best_move, _, fen = line.strip().split(maxsplit=2)
            fens.append(encode_fen(fen))
            best_moves.append(encode_move(best_move))
    return np.array(fens), np.array(best_moves)


# Test
test_fens, test_best_moves = load_data(DATA_PATH)
assert (
    test_fens[0]
    == encode_fen(
        "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    )
).all()
assert (test_best_moves[0] == encode_move("b2e2")).all()


# %% [md]
# We define the architecture of our model.


# %%
def create_model():
    inputs = tfk.Input(shape=BOARD_SIZE, dtype=tf.int32)

    x = layers.Embedding(input_dim=len(PIECE_VAL) + 1, output_dim=16)(inputs)
    x = layers.Reshape((*BOARD_SIZE, 8))(x)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(len(MOVE_ID), activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


# %% [md]
# Finally, we train and save our model.

# %%
data = load_data(DATA_PATH)

batch_size = 16
dataset = (
    tf.data.Dataset.from_tensor_slices(data)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)
dataset_size = len(data[0])
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

model = create_model()
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_dataset, epochs=10, validation_data=val_dataset, verbose=1)

model.save("xiangqi_model.keras")
