"""CNN architecture for 1D TEM (transient electromagnetic) inversion.

The network maps a sequence of dBz/dt voltage-decay measurements (one
value per time gate) to a sequence of resistivity/depth values (one pair
per subsurface layer), using a small multi-scale dilated-convolution
encoder/decoder -- similar in spirit to a 1D U-Net with an
Inception-style bottleneck.

This is the single, de-duplicated version of `Net_modified`, which
appeared twice (identically, aside from an unused default argument) in
the original notebook.
"""

from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    Reshape,
    UpSampling1D,
    concatenate,
)
from tensorflow.keras.models import Model


def build_model(im_height: int = 28, output_dim: int = 40, neurons: int = 16, kern_sz: int = 5) -> Model:
    """Build the TEM inversion CNN.

    Args:
        im_height: Number of input time gates (sequence length). Must
            match the number of columns in the preprocessed ``X`` array.
        output_dim: Width of the target vector, i.e.
            ``n_resistivity_layers + n_depth_layers`` (e.g. ``20 + 20 =
            40``). Must match the number of columns in ``y``.
        neurons: Base number of convolution filters; doubled/quadrupled/
            octupled at each downsampling stage.
        kern_sz: Convolution kernel size used throughout the
            encoder/decoder.

    Returns:
        An uncompiled Keras ``Model`` with input shape ``(im_height, 1)``
        and output shape ``(output_dim, 1)``.

    Note:
        The original notebook reused ``im_height`` for *both* the input
        sequence length and the output size (``Dense(im_height)`` /
        ``Reshape((im_height, 1))``), even though the target ``y`` array
        is actually ``2 * n_layers`` wide (resistivity + depth
        concatenated) -- a different number from the input's time-gate
        count in general. That coincides only if a dataset happens to
        have exactly ``im_height / 2`` layers. This version takes
        ``output_dim`` as its own explicit argument so the two are no
        longer silently tied together; see the README for details.
    """
    input_img = Input((im_height, 1))

    # --- Encoder ---
    conv1 = Conv1D(neurons, kernel_size=kern_sz, activation="relu", padding="same")(input_img)
    pool1 = MaxPooling1D(2, padding="same")(conv1)

    conv2 = Conv1D(neurons * 2, kernel_size=kern_sz, activation="relu", padding="same")(pool1)
    pool2 = MaxPooling1D(2, padding="same")(conv2)

    conv3 = Conv1D(neurons * 4, kernel_size=kern_sz, activation="relu", padding="same")(pool2)
    pool3 = MaxPooling1D(2, padding="same")(conv3)

    # --- Multi-scale dilated bottleneck (Inception-style) ---
    py1 = Conv1D(neurons * 8, kernel_size=kern_sz, dilation_rate=1, activation="relu", padding="same")(pool3)
    py2 = Conv1D(neurons * 8, kernel_size=kern_sz, dilation_rate=2, activation="relu", padding="same")(pool3)
    py3 = Conv1D(neurons * 8, kernel_size=kern_sz, dilation_rate=4, activation="relu", padding="same")(pool3)
    merged = concatenate([py1, py2, py3, pool3])

    bottleneck = Conv1D(neurons * 8, kernel_size=3, activation="relu", padding="same")(merged)
    up_bottleneck = UpSampling1D(2)(bottleneck)
    up_bottleneck = UpSampling1D(2)(up_bottleneck)
    up_bottleneck = UpSampling1D(2)(up_bottleneck)

    # --- Decoder ---
    deconv1 = Conv1D(neurons * 4, kernel_size=kern_sz, activation="relu", padding="same")(pool3)
    up1 = UpSampling1D(2)(deconv1)

    deconv2 = Conv1D(neurons * 2, kernel_size=kern_sz, activation="relu", padding="same")(up1)
    up2 = UpSampling1D(2)(deconv2)

    deconv3 = Conv1D(neurons, kernel_size=kern_sz, activation="relu", padding="same")(up2)
    up3 = UpSampling1D(2)(deconv3)

    merged_decoder = concatenate([up_bottleneck, up3])
    deconv4 = Conv1D(neurons, kernel_size=kern_sz, activation="relu", padding="same")(merged_decoder)
    conv_out = Conv1D(1, kernel_size=1, activation="linear")(deconv4)

    # --- Dense head ---
    x = Flatten()(conv_out)
    x = Dense(128, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(output_dim)(x)

    output = Reshape((output_dim, 1))(x)
    output = Conv1D(1, kernel_size=1, activation="linear")(output)

    return Model(inputs=input_img, outputs=output, name="tem_inversion_net")
