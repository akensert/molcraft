import tensorflow as tf 
import keras


class Attention(keras.layers.Layer):

    '''Transformer module (encoder or decoder).

    This module can be used as a transformer encoder or transformer decoder
    depending on whether causal masking is used.
    '''

    def __init__(
        self,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_masking: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.supports_masking = True
        
        self.attn = keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dense_dim, activation='relu'),
            keras.layers.Dense(embed_dim)
        ])
        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = keras.layers.Dropout(dropout)
        self.dropout_2 = keras.layers.Dropout(dropout)
        self.causal_masking = causal_masking
        

    def call(self, inputs, mask=None):

        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype='int32')
        else:
            padding_mask = None

        residual = inputs 

        attn_outputs = self.attn(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            attention_mask=padding_mask, 
            use_causal_mask=self.causal_masking)
        
        attn_outputs = self.dropout_1(attn_outputs)
        attn_outputs = self.norm_1(attn_outputs + residual)

        residual = attn_outputs 

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout_2(ffn_outputs)

        return self.norm_2(ffn_outputs + residual)
    
    
class CrossAttention(Attention):

    '''Transformer module with cross attention (decoder only).
    
    See the original transformer paper.
    '''

    def __init__(
        self,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_masking: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.attn_1 = keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.attn_2 = keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dense_dim, activation='relu'),
            keras.layers.Dense(embed_dim)
        ])
        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = keras.layers.Dropout(dropout)
        self.dropout_2 = keras.layers.Dropout(dropout)
        self.dropout_3 = keras.layers.Dropout(dropout)
        self.causal_masking = causal_masking
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):

        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype='int32')
        else:
            padding_mask = None
            
        residual = inputs 
        
        attn_outputs = self.attn_1(
            query=inputs, 
            value=inputs, 
            key=inputs, 
            use_causal_mask=self.causal_masking)
        
        attn_outputs = self.dropout_1(attn_outputs)
        attn_outputs = self.norm_1(attn_outputs + residual)

        residual = attn_outputs 

        cross_attn_outputs = self.attn_2(
            query=attn_outputs,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            use_causal_mask=self.causal_masking,
        )
        cross_attn_outputs = self.dropout_2(cross_attn_outputs)
        cross_attn_outputs = self.norm_2(cross_attn_outputs + residual)

        residual = cross_attn_outputs

        ffn_outputs = self.ffn(cross_attn_outputs)
        ffn_outputs = self.dropout_3(ffn_outputs)

        return self.norm_3(ffn_outputs + residual)