import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.keras as tfk
from bidict import bidict
from tensorflow.keras import layers, models

DATA_PATH = "data/MOVES.DAT"
BOARD_SHAPE = (10, 9, 7)

PIECE_VAL = bidict(
    {
        "K": 0,  # King
        "A": 1,  # Advisor
        "B": 2,  # Elephant
        "N": 3,  # Horse
        "R": 4,  # Chariot
        "C": 5,  # Cannon
        "P": 6,  # Pawn
    }
)

columns = "abcdefghi"
rows = range(0, 10)

COLUMN_ID = bidict({c: i for i, c in enumerate(columns)})


def encode_fen(fen: str) -> npt.NDArray[np.int_]:
    board = np.zeros(BOARD_SHAPE, dtype=int)
    rows = fen.split()[0].split("/")
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board[i, col, PIECE_VAL[char.upper()]] = 1 if char.isupper() else -1
                col += 1
    return board


def decode_fen(fen_board: np.ndarray, turn: str) -> str:
    fen_rows = []

    for row in [fen_board[i, :, :] for i in range(BOARD_SHAPE[0])]:
        fen_row = ""
        empty_count = 0

        for col in [row[j, :] for j in range(BOARD_SHAPE[1])]:
            if not np.any(col):
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                piece_type = np.where(col != 0)[0][0]
                piece_char = PIECE_VAL.inv[np.abs(piece_type)]

                if col[piece_type] == 1:
                    piece_char = piece_char.upper()
                else:
                    piece_char = piece_char.lower()

                fen_row += piece_char

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)

    turn_char = "w" if turn.lower()[0] == "r" or turn.lower() == "w" else "b"

    return f"{fen} {turn_char}"


def encode_move(move: str) -> list[npt.NDArray[np.int_]]:
    from_c, from_r = move[:2]
    one_hot_from = np.zeros(BOARD_SHAPE[:2], dtype=np.int_)
    one_hot_from[BOARD_SHAPE[0] - 1 - int(from_r), COLUMN_ID[from_c]] = 1

    to_c, to_r = move[2:]
    one_hot_to = np.zeros(BOARD_SHAPE[:2], dtype=np.int_)
    one_hot_to[BOARD_SHAPE[0] - 1 - int(to_r), COLUMN_ID[to_c]] = 1

    return [one_hot_from.flatten(), one_hot_to.flatten()]


def decode_move(encoded_move: list[npt.NDArray[np.int_]]) -> str:
    one_hot_from, one_hot_to = (
        encoded_move[0].reshape(BOARD_SHAPE[:2]),
        encoded_move[1].reshape(BOARD_SHAPE[:2]),
    )

    from_index = np.argwhere(one_hot_from == 1)[0]
    to_index = np.argwhere(one_hot_to == 1)[0]

    from_r = BOARD_SHAPE[0] - 1 - from_index[0]
    from_c = COLUMN_ID.inv[from_index[1]]

    to_r = BOARD_SHAPE[0] - 1 - to_index[0]
    to_c = COLUMN_ID.inv[to_index[1]]

    return f"{from_c}{from_r}{to_c}{to_r}"


def load_data(
    file_path: str,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    fens: list[npt.NDArray[np.int_]] = []
    from_moves: list[npt.NDArray[np.int_]] = []
    to_moves: list[npt.NDArray[np.int_]] = []
    with open(file_path, "r") as f:
        for line in f:
            best_move, _, fen = line.strip().split(maxsplit=2)
            best_move_encoded = encode_move(best_move)
            fen_encoded = encode_fen(fen)

            fens.append(fen_encoded)

            from_moves.append(best_move_encoded[0])
            to_moves.append(best_move_encoded[1])
    return np.array(fens), np.array(from_moves), np.array(to_moves)


def create_model():
    inputs = tfk.Input(shape=BOARD_SHAPE)

    from_head = layers.Conv2D(128, (3, 3), padding="same")(inputs)
    from_head = layers.BatchNormalization()(from_head)
    from_head = layers.ReLU()(from_head)
    from_head = layers.Conv2D(256, (5, 5), padding="same")(from_head)
    from_head = layers.BatchNormalization()(from_head)
    from_head = layers.ReLU()(from_head)

    from_head = layers.Flatten()(from_head)
    from_head = layers.Dense(90, activation="softmax", name="output_from_move")(
        from_head
    )

    from_head_reshaped = layers.Reshape((10, 9, 1))(from_head)

    combined = layers.Concatenate(axis=-1)([inputs, from_head_reshaped])

    to_head = layers.Conv2D(128, (3, 3), padding="same")(combined)
    to_head = layers.BatchNormalization()(to_head)
    to_head = layers.ReLU()(to_head)
    to_head = layers.Conv2D(256, (5, 5), padding="same")(to_head)
    to_head = layers.BatchNormalization()(to_head)
    to_head = layers.ReLU()(to_head)

    to_head = layers.Flatten()(to_head)
    to_head = layers.Dense(90, activation="softmax", name="output_to_move")(to_head)

    model = models.Model(
        inputs=inputs, outputs=[from_head, to_head], name="xiangqi_model"
    )
    return model


if __name__ == "__main__":
    tfk.mixed_precision.set_global_policy("mixed_float16")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    X, Y_from, Y_to = load_data(DATA_PATH)
    dataset = tf.data.Dataset.from_tensor_slices((X, (Y_from, Y_to)))
    dataset = dataset.shuffle(buffer_size=X.shape[0])

    train_size = int(0.8 * X.shape[0])
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    batch_size = 32
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    model = create_model()
    model.compile(
        optimizer="adam",
        loss={
            "output_from_move": "categorical_crossentropy",
            "output_to_move": "categorical_crossentropy",
        },
        metrics={
            "output_from_move": "accuracy",
            "output_to_move": "accuracy",
        },
    )
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=10,
        batch_size=batch_size,
    )

    model.save("xiangqi_model.keras")
