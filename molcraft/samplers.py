import tensorflow as tf
import abc
import typing 
import keras

from keras import ops
from molcraft import tokenizers
from molcraft import models


class Sampler(abc.ABC):

    def __init__(
        self,
        model: models.TransformerDecoder,
        tokenizer: tokenizers.Tokenizer,
        temperature: float = 1.0,
        run_eagerly: bool = False,
    ) -> None:
        if not (tokenizer.bos_token and tokenizer.eos_token):
            raise ValueError(
                'Tokenizer requires start and stop token. '
                'Specify `add_bos=True` and `add_eos=True` for tokenizer.'
            )
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.run_eagerly = run_eagerly
        if not self.run_eagerly:
            self._prepare = tf.function(self._prepare)
            self._generate = tf.function(self._generate)
    
    def sample(
        self,
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor | None = None,
        **kwargs,
    ) -> tf.Tensor:
        if not tf.is_tensor(decoder_sequence):
            decoder_sequence = ops.convert_to_tensor(decoder_sequence)
            if decoder_sequence.dtype != tf.string:
                raise ValueError(
                    'The dtype of `decoder_sequence` needs to tf.string.'
                )
        if encoder_sequence is not None and not tf.is_tensor(encoder_sequence):
            encoder_sequence = ops.convert_to_tensor(encoder_sequence)
            if encoder_sequence.dtype != tf.string:
                raise ValueError(
                    'The dtype of `encoder_sequence` needs to tf.string.'
                )
        return self(decoder_sequence, encoder_sequence, **kwargs) 

    def __call__(
        self,
        decoder_input: tf.Tensor,
        encoder_output: tf.Tensor | None = None,
        **kwargs      
    ) -> tf.Tensor:
        last_token, decoder_input, *loop_vars = self._prepare(
            decoder_input, encoder_output
        )

        loop_vars = ((), (), *loop_vars)
        tokens, _, _ = self._generate(last_token, *loop_vars)


        decoder_output = keras.ops.concatenate([
            decoder_input, tokens
        ], axis=1)

        return self.tokenizer.detokenize(decoder_output)

    def _prepare(
        self,
        decoder_input: tf.Tensor,
        encoder_output: tf.Tensor | None = None,
    ):
        batch_size = ops.shape(decoder_input)[0]
        sequence_length = self.model.sequence_length
        
        decoder_input = self.tokenizer.tokenize(
            decoder_input, ragged=True)
        decoder_input = decoder_input[:, :-1]
        sequence_lengths = decoder_input.row_lengths()

        self_attention_cache = self.model.initialize_attention_cache(
            batch_size, sequence_length
        )
        self_attention_cache_update_index = tf.ragged.range(sequence_lengths)
        if encoder_output is not None:
            cross_attention_cache = self.model.initialize_attention_cache(
                batch_size, sequence_length
            )
            cross_attention_cache_update_index = tf.ragged.range(sequence_lengths)
        else:
            cross_attention_cache = cross_attention_cache_update_index = None

        current_token = decoder_input[:, -1:].to_tensor()[:, 0]

        _, attention_cache = self.model(
            decoder_sequence=self.tokenizer.pad(
                decoder_input, sequence_length=None
            ),
            encoder_output=encoder_output,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
            cross_attention_cache=cross_attention_cache,
            cross_attention_cache_update_index=cross_attention_cache_update_index,
        )

        if encoder_output is not None:
            self_attention_cache, cross_attention_cache = attention_cache
        else:
            self_attention_cache = attention_cache

        self_attention_cache_update_index = ops.zeros(
            shape=(batch_size, 1), dtype='int32'
        )

        return (
            current_token,
            decoder_input,
            self_attention_cache,
            self_attention_cache_update_index,
            encoder_output,
            cross_attention_cache,
            cross_attention_cache_update_index,
        ) 

    def _generate(
        self, 
        current_token: tf.TensorArray,
        log_probs: typing.Optional[tf.TensorArray],
        entropies: typing.Optional[tf.TensorArray],
        self_attention_cache: typing.Optional[tf.Tensor],
        self_attention_cache_update_index: typing.Optional[tf.Tensor],
        encoder_output: typing.Optional[tf.Tensor],
        cross_attention_cache: typing.Optional[tf.Tensor],
        cross_attention_cache_update_index: typing.Optional[tf.Tensor],
    ):
        
        def _cond(
            tokens: tf.TensorArray,
            *_,
        ) -> bool:
            stop_token_id = self.tokenizer.eos_token_id
            return keras.ops.logical_not(
                keras.ops.all(
                    keras.ops.any(tokens.stack() == stop_token_id, axis=0)
                )
            )
    
        def _step(
            tokens,
            log_probs,
            entropies,
            self_attention_cache,
            self_attention_cache_update_index,
            encoder_output,
            cross_attention_cache,
            cross_attention_cache_update_index,
        ):
            index = tokens.size()
            current_token = tokens.read(index - 1)
            current_token = ops.expand_dims(current_token, axis=1)

            outputs = self.model(
                decoder_sequence=current_token,
                self_attention_cache=(
                    self_attention_cache if tf.is_tensor(
                        self_attention_cache
                    ) else None
                ),
                self_attention_cache_update_index=(
                    self_attention_cache_update_index if tf.is_tensor(
                        self_attention_cache_update_index
                    ) else None
                ),
                encoder_output=(
                    encoder_output if tf.is_tensor(
                        encoder_output
                    ) else None
                ),
                cross_attention_cache=(
                    cross_attention_cache if tf.is_tensor(
                        cross_attention_cache
                    ) else None
                ),
                cross_attention_cache_update_index=(
                    cross_attention_cache_update_index if tf.is_tensor(
                        cross_attention_cache_update_index
                    ) else None
                ),
            )
            if not isinstance(outputs, (tuple, list)):
                logits = outputs 
            elif len(outputs) == 2:
                logits, self_attention_cache = outputs
            else:
                logits, self_attention_cache, cross_attention_cache = outputs
   
            logits = ops.squeeze(logits, axis=1)

            next_token = self.get_next_token(logits)
            
            next_token = ops.squeeze(next_token, axis=1)
            tokens = tokens.write(index, next_token)

            log_prob = ops.log_softmax(logits)

            if isinstance(log_probs, tf.TensorArray):
                log_probs = log_probs.write(
                    index, _log_prob(next_token, log_prob)
                )
            
            if isinstance(entropies, tf.TensorArray):
                entropies = entropies.write(
                    index, _entropy(log_prob)
                )

            if tf.is_tensor(self_attention_cache_update_index):
                self_attention_cache_update_index += 1
            
            if tf.is_tensor(cross_attention_cache_update_index):
                cross_attention_cache_update_index += 1

            return (
                tokens,
                log_probs,
                entropies,
                self_attention_cache,
                self_attention_cache_update_index,
                encoder_output,
                cross_attention_cache,
                cross_attention_cache_update_index,
            )

        tokens = tf.TensorArray(
            dtype='int32', size=0, dynamic_size=True, clear_after_read=False
        )
        tokens = tokens.write(0, current_token)

        tokens, log_probs, entropies, *_ = ops.while_loop(
            _cond, _step, loop_vars=(
                tokens,
                log_probs,
                entropies,
                self_attention_cache,
                self_attention_cache_update_index,
                encoder_output or (),
                cross_attention_cache or (),
                cross_attention_cache_update_index or (),
            )
        )

        tokens = _unpack_array(tokens)
        log_probs = _unpack_array(log_probs)
        entropies = _unpack_array(entropies)

        return tokens, log_probs, entropies

    def get_next_token(self, logits: tf.Tensor) -> tf.Tensor:
        return tf.random.categorical(
            logits=logits / self.temperature,
            num_samples=1,
            dtype='int32'
        )


class TopKSampler(Sampler):

    def __init__(
        self,
        model: keras.models.Model,
        tokenizer: tokenizers.Tokenizer,
        k: int = 5,
        temperature: float = 0.7,
        run_eagerly: bool = False,
    ) -> None:
        super().__init__(
            model=model, 
            tokenizer=tokenizer, 
            temperature=temperature,
            run_eagerly=run_eagerly,
        )
        self.k = k

    def get_next_token(self, logits: tf.Tensor) -> tf.Tensor:
        logits, indices = ops.top_k(logits, self.k, sorted=False)
        sample = super().get_next_token(logits)
        return ops.take_along_axis(indices, sample, axis=-1)
    
    
class _GreedySampler(Sampler):
    pass


class _TopPSampler(Sampler):
    pass 


class _BeamSampler(Sampler):
    pass


def _log_prob(token: tf.Tensor, log_prob: tf.Tensor) -> tf.Tensor:
    return ops.take_along_axis(log_prob, token, axis=1)

def _entropy(log_prob: tf.Tensor) -> tf.Tensor:
    prob = tf.math.exp(log_prob)
    log_prob = ops.where(prob == 0, ops.zeros_like(log_prob), log_prob)
    return -ops.sum(log_prob * prob, axis=-1)

def _unpack_array(array: tf.TensorArray) -> tf.Tensor:
    if isinstance(array, tf.TensorArray):
        return ops.transpose(array.stack())
    return array