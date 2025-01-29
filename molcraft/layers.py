import tensorflow as tf
import keras


class TokenEmbedding(keras.layers.Layer):

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        initializer: keras.initializers.Initializer = "glorot_uniform",
        mask_zero: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.initializer = keras.initializers.get(initializer)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "embedding_dim": self.embedding_dim,
                "initializer": keras.initializers.serialize(self.initializer),
                "mask_zero": self.mask_zero,
            }
        )
        return config
    
    def build(self, inputs_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.vocabulary_size, self.embedding_dim],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, sequence: tf.Tensor, start_index: int = 0):
        if sequence.dtype != "int32" and sequence.dtype != "int64":
            sequence = keras.ops.cast(sequence, "int32")
        outputs = keras.ops.take(self.embeddings, sequence, axis=0)
        return keras.ops.cast(outputs, dtype=self.compute_dtype)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return keras.ops.not_equal(inputs, 0)
    
    def compute_output_shape(self, input_shape):
        return (*input_shape, self.embedding_dim)


class PositionEmbedding(keras.layers.Layer):

    def __init__(
        self,
        sequence_length: int,
        initializer: keras.initializers.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config
    
    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, sequence, start_index=0):
        sequence_shape = keras.ops.shape(sequence)
        sequence_length = sequence_shape[-2]
        feature_length = sequence_shape[-1]
        embeddings = keras.ops.convert_to_tensor(self.embeddings)
        embeddings = keras.ops.slice(
            embeddings,
            start_indices=(start_index, 0),
            shape=(sequence_length, feature_length),
        )
        return keras.ops.broadcast_to(embeddings, sequence_shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class SegmentEmbedding(keras.layers.Layer):

    def __init__(
        self,
        num_segments: int,
        initializer: keras.initializers.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_segments = int(num_segments)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_segments": self.num_segments,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config
    
    def build(self, inputs_shape):
        feature_size = inputs_shape[-1]
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.num_segments, feature_size],
            initializer=self.initializer,
            trainable=True,
        )
        self.built = True

    def call(self, sequence, start_index=0):
        # sequence_shape = keras.ops.shape(sequence)
        pass

    def compute_output_shape(self, input_shape):
        return input_shape