from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (LSTM,  # type: ignore
                                     Input,
                                     RepeatVector,
                                     TimeDistributed,
                                     Dense)
from tensorflow.keras.optimizers import Adam  # type: ignore


def get_autoencoder(input_data=None, input_shape=None, learning_rate=0.001, encoder_seq=None):
    """
    Constructs an LSTM Autoencoder with dynamic layer sizing and learning rate.

    The model consists of an Encoder (reducing dimensions via LSTM), a Latent Space
    (RepeatVector), and a Decoder (reconstructing dimensions via LSTM).

    Args:
        input_data (np.ndarray, optional): Sample data to infer shape from.
        input_shape (tuple, optional): Explicit shape tuple (steps, features) or (batch, steps, features).
        learning_rate (float): Learning rate for the Adam optimizer.
        encoder_seq (list[int], optional): List defining the number of units in each encoder LSTM layer.
                                           Defaults to [64, 32].

    Returns:
        Model: A compiled Keras functional Model ready for training.

    Raises:
        ValueError: If neither input_data nor input_shape is provided, or if shape is invalid.
    """
    if encoder_seq is None:
        encoder_seq = [64, 32]

    if input_data is not None:
        input_shape = input_data.shape

    # Robust shape handling
    if input_shape is None:
        raise ValueError("Must provide either input_data or input_shape")

    if len(input_shape) == 2:
        steps = input_shape[0]
        features = input_shape[1]
    elif len(input_shape) == 3:
        # (batch, steps, features) or (None, steps, features)
        steps = input_shape[1]
        features = input_shape[2]
    else:
        raise ValueError(f"Unexpected input_shape: {input_shape}")

    # --- 1. Encoder ---
    input_layer = Input(shape=(steps, features))
    x = input_layer

    # Dynamically build encoder layers
    # Example: if encoder_seq=[64, 32], we add LSTM(64) then LSTM(32)
    for i, units in enumerate(encoder_seq):
        # Last layer of encoder must NOT return sequences to feed into RepeatVector
        is_last_layer = (i == len(encoder_seq) - 1)
        x = LSTM(units, return_sequences=not is_last_layer)(x)

    # --- 2. Latent Space ---
    x = RepeatVector(steps)(x)

    # --- 3. Decoder ---
    # We mirror the encoder sequence for the decoder
    decoder_seq = encoder_seq[::-1]  # Reverse it: [32, 64]

    for units in decoder_seq:
        x = LSTM(units, return_sequences=True)(x)

    output_layer = TimeDistributed(Dense(features, activation='linear'))(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    # --- 4. Compilation ---
    optimizer = Adam(learning_rate=learning_rate)

    # run_eagerly=True is safer for older CPUs/thread conflicts
    model.compile(loss="mse", optimizer=optimizer, run_eagerly=True)

    return model
