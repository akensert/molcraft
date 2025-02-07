import tensorflow as tf
import keras
import typing

from molcraft import layers 


class TransformerDecoder(keras.Model):
 
    def __init__(
        self,         
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        intermediate_dim: int,
        vocabulary_size: int,
        sequence_length: int,
        dropout: float = 0.0,
        activation: keras.layers.Activation = "relu",
        layer_norm_epsilon: float = 1e-05,
        kernel_initializer: keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: keras.initializers.Initializer = "zeros",
        embeddings_initializer: keras.initializers.Initializer = "uniform",
        normalize_first: bool = False,
        mask_zero: bool = True,
        use_causal_mask: bool = True,
        name: str = 'GenerativeTransformer',
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_units = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.embeddings_initializer = keras.initializers.get(
            embeddings_initializer
        )
        self.normalize_first = normalize_first
        self.mask_zero = mask_zero
        self.use_causal_mask = use_causal_mask

        self.token_embedding = layers.TokenEmbedding(
            self.vocabulary_size,
            self.embedding_dim,
            initializer=layers.clone_initializer(
                self.embeddings_initializer
            ),
            mask_zero=self.mask_zero,
        )

        self.position_embedding = layers.PositionEmbedding(
            self.sequence_length,
            initializer=layers.clone_initializer(
                self.embeddings_initializer
            )
        )

        self.decoder_layers = [
            layers.TransformerDecoderLayer(
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_epsilon=self.layer_norm_epsilon,
                kernel_initializer=layers.clone_initializer(
                    self.kernel_initializer
                ),
                bias_initializer=layers.clone_initializer(
                    self.bias_initializer
                ),
                normalize_first=self.normalize_first,
            )
            for _ in range(self.num_layers)
        ]

        self.dense = keras.layers.Dense(self.vocabulary_size)

    @property 
    def num_layers(self):
        return self._num_layers 
    
    @property 
    def num_heads(self):
        return self._num_heads 
    
    @property 
    def num_units(self):
        return self._num_units
    
    def call(
        self, 
        decoder_sequence: tf.Tensor, 
        self_attention_cache: typing.Optional[tf.Tensor] = None,
        self_attention_cache_update_index: typing.Optional[tf.Tensor] = None,
        encoder_output: typing.Optional[tf.Tensor] = None,
        cross_attention_cache: typing.Optional[tf.Tensor] = None,
        cross_attention_cache_update_index: typing.Optional[tf.Tensor] = None,
        **kwargs
    ) -> typing.Union[
        tf.Tensor, 
        typing.Tuple[tf.Tensor, tf.Tensor],
        typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ]:  
        self_attention_caching = (self_attention_cache is not None) 
        cross_attention_caching = (cross_attention_cache is not None) 

        embedding = self.token_embedding(decoder_sequence)
        embedding += self.position_embedding(
            embedding, index=self_attention_cache_update_index)

        x = embedding

        if self_attention_caching:
            updated_self_attention_cache = []

        if cross_attention_caching:
            updated_cross_attention_cache = []

        for i, decoder in enumerate(self.decoder_layers):
            output = decoder(
                x,
                self_attention_cache=(
                    None if self_attention_cache is None else
                    self_attention_cache[:, i, :, :, :, :]
                ),
                self_attention_cache_update_index=(
                    self_attention_cache_update_index
                ),
                encoder_sequence=encoder_output,
                cross_attention_cache=(
                    None if cross_attention_cache is None else
                    cross_attention_cache[:, i, :, :, :, :]
                ),
                cross_attention_cache_update_index=(
                    cross_attention_cache_update_index
                ),
                use_causal_mask=self.use_causal_mask,
                **kwargs
            )

            if isinstance(output, (tuple, list)):
                x, *updated_attention_cache = output
            else:
                x = output

            if self_attention_caching:
                updated_self_attention_cache.append(updated_attention_cache[0])

            if cross_attention_caching:
                updated_cross_attention_cache.append(updated_attention_cache[1])

        if self_attention_caching:
            self_attention_cache = keras.ops.stack(
                updated_self_attention_cache, axis=1)
        
        if cross_attention_caching:
            cross_attention_cache = keras.ops.stack(
                updated_cross_attention_cache, axis=1)

        x = self.dense(x)

        if not self_attention_caching and not cross_attention_caching:
            return x
        
        if self_attention_caching and not cross_attention_caching:
            return x, self_attention_cache
        
        return x, self_attention_cache, cross_attention_cache

    def initialize_attention_cache(
        self, 
        batch_size: int,
        sequence_length: int,
        dtype: str = 'float32',
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return keras.ops.zeros(
            shape=(
                batch_size,
                self.num_layers,
                2,
                sequence_length,
                self.num_heads,
                self.num_units
            ),
            dtype=dtype
        )