import tensorflow as tf
import keras


class TransformerDecoderLayer(keras.layers.Layer):

    def __init__(
        self,
        intermediate_dim: int,
        num_heads: int,
        dropout: float = 0,
        activation: keras.layers.Activation = "relu",
        layer_norm_epsilon: float = 1e-05,
        kernel_initializer: keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: keras.initializers.Initializer = "zeros",
        normalize_first: bool = False,
        **kwargs,
    ) -> None:
        decoder_sequence_shape = kwargs.pop("decoder_sequence_shape", None)
        encoder_sequence_shape = kwargs.pop("encoder_sequence_shape", None)

        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True
        self._decoder_sequence_shape = None
        self._encoder_sequence_shape = None

        if decoder_sequence_shape:
            self.build(decoder_sequence_shape, encoder_sequence_shape)
    
    def build(
        self,
        decoder_sequence_shape: tf.TensorShape,
        encoder_sequence_shape: tf.TensorShape | None = None,
    ) -> None:
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        hidden_dim = decoder_sequence_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)

        self.self_attention_layer = KeyValueCachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        if hasattr(self.self_attention_layer, "_build_from_signature"):
            self.self_attention_layer._build_from_signature(
                query=decoder_sequence_shape,
                value=decoder_sequence_shape,
            )
        else:
            self.self_attention_layer.build(
                query_shape=decoder_sequence_shape,
                value_shape=decoder_sequence_shape,
            )
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self.self_attention_layer_norm.build(decoder_sequence_shape)
        self.self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        self.cross_attention_layer = None
        if encoder_sequence_shape:
            self.cross_attention_layer = KeyValueCachedMultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                dropout=self.dropout,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                dtype=self.dtype_policy,
                name="cross_attention",
            )
            if hasattr(self.cross_attention_layer, "_build_from_signature"):
                self.cross_attention_layer._build_from_signature(
                    query=decoder_sequence_shape,
                    value=encoder_sequence_shape,
                )
            else:
                self.cross_attention_layer.build(
                    query_shape=decoder_sequence_shape,
                    value_shape=encoder_sequence_shape,
                )
            self.cross_attention_layer_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="cross_attention_layer_norm",
            )
            self.cross_attention_layer_norm.build(decoder_sequence_shape)
            self.cross_attention_dropout = keras.layers.Dropout(
                rate=self.dropout,
                dtype=self.dtype_policy,
                name="cross_attention_dropout",
            )

        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(decoder_sequence_shape)
        self.feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.feedforward_output_dense.build(tuple(intermediate_shape))
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self.feedforward_layer_norm.build(decoder_sequence_shape)
        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(
        self,
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor = None,
        decoder_padding_mask: tf.Tensor = None,
        decoder_attention_mask: tf.Tensor = None,
        encoder_padding_mask: tf.Tensor = None,
        encoder_attention_mask: tf.Tensor = None,
        self_attention_cache: tf.Tensor = None,
        self_attention_cache_update_index: tf.Tensor = None,
        cross_attention_cache: tf.Tensor = None,
        cross_attention_cache_update_index: tf.Tensor = None,
        use_causal_mask: bool = True,
        training: bool = None,
    ):

        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            use_causal_mask=use_causal_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )
        x = decoder_sequence
        residual = x
        if self.normalize_first:
            x = self.self_attention_layer_norm(x)
        attention_output = self.self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        if self_attention_cache is None:
            x = attention_output
        else:
            x, self_attention_cache = attention_output
        x = self.self_attention_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self.self_attention_layer_norm(x)

        if self.cross_attention_layer is not None:
            cross_attention_mask = _merge_padding_and_attention_mask(
                encoder_sequence, encoder_padding_mask, encoder_attention_mask
            )
            residual = x
            if self.normalize_first:
                x = self.cross_attention_layer_norm(x)
            attention_output = self.cross_attention_layer(
                query=x,
                value=encoder_sequence,
                attention_mask=cross_attention_mask,
                cache=cross_attention_cache,
                cache_update_index=cross_attention_cache_update_index,
                training=training,
            )
            if cross_attention_cache is None:
                x = attention_output
            else:
                x, cross_attention_cache = attention_output
            x = self.cross_attention_dropout(x, training=training)
            x = x + residual
            if not self.normalize_first:
                x = self.cross_attention_layer_norm(x)

        residual = x
        if self.normalize_first:
            x = self.feedforward_layer_norm(x)
        x = self.feedforward_intermediate_dense(x)
        x = self.feedforward_output_dense(x)
        x = self.feedforward_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self.feedforward_layer_norm(x)

        if self_attention_cache is not None:
            if self.cross_attention_layer is not None:
                return (x, self_attention_cache, cross_attention_cache)
            else:
                return (x, self_attention_cache)
        else:
            return x
        
    def _compute_self_attention_mask(
        self,
        decoder_sequence: tf.Tensor,
        decoder_padding_mask: tf.Tensor,
        decoder_attention_mask: tf.Tensor,
        use_causal_mask: bool,
        self_attention_cache: tf.Tensor,
        self_attention_cache_update_index: tf.Tensor,
    ) -> tf.Tensor:
        decoder_mask = _merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if use_causal_mask:
            batch_size = keras.ops.shape(decoder_sequence)[0]
            input_length = output_length = keras.ops.shape(decoder_sequence)[1]
            if self_attention_cache is not None:
                input_length = keras.ops.shape(self_attention_cache)[2]
            causal_mask = _compute_causal_mask(
                batch_size,
                input_length,
                output_length,
                self_attention_cache_update_index,
            )
            return (
                keras.ops.minimum(decoder_mask, causal_mask)
                if decoder_mask is not None
                else causal_mask
            )
        return decoder_mask

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "intermediate_dim": self.intermediate_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": keras.activations.serialize(self.activation),
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "normalize_first": self.normalize_first,
            "decoder_sequence_shape": self._decoder_sequence_shape,
            "encoder_sequence_shape": self._encoder_sequence_shape,
        })
        return config

    def compute_output_shape(
        self, 
        decoder_sequence_shape: tf.TensorShape
    ) -> tf.TensorShape:
        return decoder_sequence_shape
    

class KeyValueCachedMultiHeadAttention(keras.layers.MultiHeadAttention):

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        if key is None:
            key = value

        query = self._query_dense(query)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self._key_dense(key)
                value_update = self._value_dense(value)
                key = _cache_update(
                    key_cache, cache_update_index, key_update
                )
                value = _cache_update(
                    value_cache, cache_update_index, value_update
                )
                cache = keras.ops.stack((key, value), axis=1)
        else:
            key = self._key_dense(key)
            value = self._value_dense(value)

        attention_output, unused_attention_scores = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            training=training,
        )

        attention_output = self._output_dense(attention_output)

        if cache is not None:
            return attention_output, cache
        return attention_output


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

    def call(self, sequence: tf.Tensor):
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

    def call(self, sequence, index: tf.Tensor = None):
        embeddings = keras.ops.convert_to_tensor(self.embeddings)
        if index is None:
            sequence_shape = keras.ops.shape(sequence)
            sequence_length = sequence_shape[-2]
            feature_length = sequence_shape[-1]
            embeddings = keras.ops.slice(
                embeddings,
                start_indices=(0, 0),
                shape=(sequence_length, feature_length),
            )
            return keras.ops.broadcast_to(embeddings, sequence_shape)
        else:
            if isinstance(index, tf.RaggedTensor):
                index = index.to_tensor()
            embeddings = keras.ops.take_along_axis(
                embeddings,
                index,
                axis=0
            )
            return keras.ops.expand_dims(embeddings, 1)
        
    def compute_output_shape(self, input_shape):
        return input_shape


def clone_initializer(initializer):
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)

def _compute_causal_mask(
    batch_size: int,
    input_length: int,
    output_length: int,
    cache_index: tf.Tensor | tf.RaggedTensor,
) -> tf.Tensor:
    if cache_index is None:
        i = keras.ops.zeros((batch_size, 1), dtype='int32')
        i += keras.ops.arange(output_length, dtype=i.dtype)
    else:
        if isinstance(cache_index, tf.RaggedTensor):
            cache_index = cache_index.to_tensor(default_value=-1)
        i = cache_index
    j = keras.ops.arange(input_length, dtype=i.dtype)
    return i[..., None] >= j[None, None, :]

def _merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):    
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            mask = inputs._keras_mask
    if mask is not None:
        mask = keras.ops.cast(keras.ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = keras.ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        else:
            return keras.ops.minimum(mask, attention_mask)
    return mask

def _cache_update(
    cache: tf.Tensor, 
    index: tf.Tensor | tf.RaggedTensor, 
    update: tf.Tensor
):
    if isinstance(index, tf.Tensor):
        index = tf.RaggedTensor.from_tensor(index)
    
    gather_index = keras.ops.stack([
        index.value_rowids(), 
        tf.ragged.range(index.row_lengths()).flat_values
    ], axis=1)
    scatter_index = keras.ops.stack([
        index.value_rowids(), 
        index.flat_values
    ], axis=1)

    update = tf.gather_nd(update, gather_index)

    return keras.ops.scatter_update(cache, scatter_index, update)