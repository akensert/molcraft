import tensorflow as tf
import keras

from keras import ops 


class SequenceEmbedding(keras.layers.Layer):

    def __init__(
        self, 
        embedding_dim: int,
        sequence_length: int,
        vocabulary_size: int,
        mask_zero: bool = True,
        learnable_positional_embedding: bool = True, 
        **kwargs
    ):
        super().__init__(**kwargs)

        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocabulary_size, output_dim=embedding_dim
        )
        if learnable_positional_embedding:
            self.positional_embedding = keras.layers.Embedding(
                input_dim=sequence_length, output_dim=embedding_dim
            )
        else:
            self.positional_embedding = SineCosinePositionalEncoding(
                output_dim=embedding_dim,
            )

        self.learnable_positional_embedding = learnable_positional_embedding
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.positional_embedding(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return ops.not_equal(inputs, 0)
    

class SineCosinePositionalEncoding(keras.layers.Layer):

    def __init__(
        self,
        output_dim: int,
        max_wavelength:int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.max_wavelength = max_wavelength
        self.built = True

    def call(self, positions: tf.Tensor) -> tf.Tensor:
        seq_length = ops.shape(positions)[0]
        positions = ops.arange(seq_length)
        positions = ops.cast(positions, self.compute_dtype)
        min_freq = ops.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = ops.power(
            min_freq,
            ops.cast(2 * (ops.arange(self.output_dim) // 2), self.compute_dtype)
            / ops.cast(self.output_dim, self.compute_dtype),
        )
        angles = ops.expand_dims(positions, 1) * ops.expand_dims(timescales, 0)
        cos_mask = ops.cast(ops.arange(self.output_dim) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        positional_encodings = (
            ops.sin(angles) * sin_mask + ops.cos(angles) * cos_mask
        )
        return positional_encodings