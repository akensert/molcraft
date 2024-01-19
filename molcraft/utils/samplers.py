import tensorflow as tf
import keras
from keras import ops 

from molcraft.definitions import SpecialTokens
from molcraft.utils.tokenizers import SMILESTokenizer


class BeamSampler(tf.Module):
    pass


class Sampler(tf.Module):

    def __init__(
        self,
        model: keras.Model,
        tokenizer: SMILESTokenizer,
        temperature: float = 0.7,
        top_k: int = None,
        id_dtype: tf.dtypes.DType = tf.int32,
        sample_eagerly: bool = False,
        name: str = 'Sampler',
    ):
        super().__init__(name=name)
        self._model = model
        self._tokenizer = tokenizer 
        self._temperature = temperature
        self._top_k = top_k
        self._pad_token_id = self._tokenizer._mask_token_id
        self._eos_token_id = self._tokenizer._eos_token_id
        self._max_sequence_length = self._tokenizer.max_sequence_length
        self._id_dtype = id_dtype
        self._sample_eagerly = sample_eagerly
        if not self._sample_eagerly:
            self.sample = tf.function(self.sample)
        
    def sample(self, input_sequences: tf.Tensor) -> tf.Tensor:

        input_sequences = self._tokenizer.tokenize(input_sequences)

        input_sequences = tf.where(
            input_sequences == self._eos_token_id, 
            self._pad_token_id, 
            input_sequences
        )
        
        pad_tokens = tf.where(input_sequences == self._pad_token_id)[:, 0]

        target_positions = (
            self._max_sequence_length 
            - tf.math.bincount(pad_tokens)
            - 1
        )
        batch_size = keras.ops.shape(input_sequences)[0]

        def cond(
            sequences: tf.Tensor, 
            indices: tf.Tensor
        ) -> bool:
            last_token = tf.gather(sequences, indices, batch_dims=1)
            return tf.reduce_any(
                tf.logical_and(
                    (indices + 1) < self._max_sequence_length,
                    last_token != self._eos_token_id
                )
            )

        def body(
            sequences: tf.Tensor, 
            indices: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            
            last_token = tf.gather(sequences, indices, batch_dims=1)

            logits = tf.gather(self._model(sequences), indices, batch_dims=1)
    
            if self._top_k:
                logits, top_k_indices = ops.top_k(
                    logits, k=self._top_k, sorted=False)
            
            sampled_token = tf.random.categorical(
                logits=ops.log_softmax(logits / self._temperature),
                num_samples=1,
                dtype=self._id_dtype
            )
    
            if self._top_k:
                sampled_token = ops.take_along_axis(
                    top_k_indices, sampled_token, axis=-1)
    
            indices_nd = tf.stack([tf.range(batch_size), indices + 1], axis=1)
            updates = tf.squeeze(sampled_token, 1)

            mask = tf.logical_and(
                indices < self._max_sequence_length,
                last_token != self._eos_token_id
            )
            updates = tf.boolean_mask(updates, mask)
            indices_nd = tf.boolean_mask(indices_nd, mask)
    
            sequences = tf.tensor_scatter_nd_update(
                sequences, indices_nd, updates
            )
            indices = tf.tensor_scatter_nd_add(
                indices, indices_nd[:, :1], tf.ones_like(indices_nd[:, 0])
            )
            
            return sequences, indices
        
        sequences, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=[input_sequences, target_positions]
        )
    
        return self._tokenizer.detokenize(sequences)

    @property
    def temperature(self):
        return self._temperature 
    
    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def top_k(self):
        return self._top_k
    
    @top_k.setter
    def top_k(self, k):
        self._top_k = k