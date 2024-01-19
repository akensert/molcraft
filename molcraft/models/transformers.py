import tensorflow as tf

import keras 

from keras import ops 

from molcraft.layers.embedding import SequenceEmbedding
from molcraft.layers.attention import Attention

def GPT(
    num_layers,
    num_heads,
    embedding_dim,
    dense_dim,
    vocabulary_size,
    sequence_length,
    dropout,
):
    inputs = keras.layers.Input(shape=(None,), dtype='int64', name='inputs')

    embedded_inputs = SequenceEmbedding(
        sequence_length=sequence_length,
        vocabulary_size=vocabulary_size,
        embedding_dim=embedding_dim,
        learnable_positional_embedding=False)(inputs)
    
    attn_outputs = embedded_inputs
    for i in range(num_layers):
        attn_outputs = Attention(
            embed_dim=embedding_dim,
            dense_dim=dense_dim,
            num_heads=num_heads,
            dropout=dropout,
            causal_masking=True,
        )(attn_outputs)

    outputs = keras.layers.Dense(vocabulary_size)(attn_outputs)
    return keras.Model(inputs, outputs)