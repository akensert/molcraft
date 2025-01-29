import keras
import keras_hub 

from molcraft import layers 


DEFAULT_SEQUENCE_LENGTH = 1_000


class TransformerDecoder(keras.Model):

    def __init__(
        self,         
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        intermediate_dim: int,
        vocabulary_size: int,
        sequence_length: int = None,
        num_segments: int = None,
        dropout: float = 0.0,
        activation: keras.layers.Activation = "relu",
        layer_norm_epsilon: float = 1e-05,
        kernel_initializer: keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: keras.initializers.Initializer = "zeros",
        embeddings_initializer: keras.initializers.Initializer = "uniform",
        normalize_first: bool = False,
        mask_zero: bool = True,
        use_causal_mask: bool = True,
        name: str = 'TransformerDecoder',
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length or DEFAULT_SEQUENCE_LENGTH
        self.num_segments = num_segments
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
            initializer=clone_initializer(self.embeddings_initializer),
            mask_zero=self.mask_zero,
        )
        self.position_embedding = layers.PositionEmbedding(
            self.sequence_length,
            initializer=clone_initializer(self.embeddings_initializer)
        )

        if self.num_segments:
            self.segment_embedding = layers.SegmentEmbedding(
                self.num_segments,
                initializer=clone_initializer(self.embeddings_initializer)
            )

        self.decoders = [
            keras_hub.layers.TransformerDecoder(
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_epsilon=self.layer_norm_epsilon,
                kernel_initializer=self.kernel_initializer.from_config(
                    self.kernel_initializer.get_config()),
                bias_initializer=self.bias_initializer.from_config(
                    self.bias_initializer.get_config()),
                normalize_first=self.normalize_first,
            )
            for _ in range(self.num_layers)
        ]
    
    def call(self, decoder_sequence, encoder_sequence=None, **kwargs):
        embedding = self.token_embedding(decoder_sequence)
        embedding += self.position_embedding(embedding)
        if self.num_segments:
            embedding += self.segment_embedding(embedding)
        x = embedding
        for decoder in self.decoders:
            x = decoder(
                decoder_sequence=x,
                encoder_sequence=encoder_sequence,
                use_causal_mask=self.use_causal_mask,
                **kwargs
            )
        return x
    

def clone_initializer(initializer):
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)