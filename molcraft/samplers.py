import tensorflow as tf
import keras 
import abc

from molcraft import tokenizers


class Sampler(abc.ABC):

    def __init__(
        self,
        model: keras.Model,
        tokenizer: tokenizers.Tokenizer,
        run_eagerly: bool = False,
    ) -> None:
        assert tokenizer.bos_token is not None, (
            'Tokenizer needs to add an `bos_token`.'
        )
        assert tokenizer.eos_token is not None, (
            'Tokenizer needs to add an `eos_token`.'
        )
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = int(tokenizer.token_to_id(tokenizer.pad_token))
        self.eos_token_id = int(tokenizer.token_to_id(tokenizer.eos_token))
        self.run_eagerly = run_eagerly
        if not self.run_eagerly:
            self.sample_fn = tf.function(self.sample_fn)

    @abc.abstractmethod
    def __call__(
        self,
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor = None,
        **kwargs,
    ) -> tf.Tensor:
        '''Sampling a complete sequence of token ids from the model.
        
        Should accept a batch of sequences, and may include an encoder sequence
        as additional input if an encoder-decoder transformer is used.
        '''

    def sample_fn(
        self,
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor = None, 
        **kwargs,
    ) -> tf.Tensor:
        decoder_sequence = self.tokenizer.tokenize(decoder_sequence)
        decoder_sequence = keras.ops.where(
            decoder_sequence == self.eos_token_id, 
            self.pad_token_id, 
            decoder_sequence
        )
        if encoder_sequence is not None:
            encoder_sequence = self.tokenizer.tokenize(encoder_sequence)
        if 'max_iter' not in kwargs and self.tokenizer.sequence_length:
            kwargs['max_iter'] = self.tokenizer.sequence_length - 1
        if 'index' not in kwargs:
            kwargs['index'] = self._target_index(decoder_sequence)
        completed_decoder_sequence = self(
            decoder_sequence, encoder_sequence, **kwargs
        )
        return self.tokenizer.detokenize(completed_decoder_sequence)
    
    def sample(
        self, 
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor = None, 
        **kwargs
    ) -> tf.Tensor:
        if not keras.ops.is_tensor(decoder_sequence):
            decoder_sequence = keras.ops.convert_to_tensor(decoder_sequence)
            if encoder_sequence is not None:
                encoder_sequence = keras.ops.convert_to_tensor(
                    encoder_sequence
                )
        sequences = self.sample_fn(decoder_sequence, encoder_sequence, **kwargs)
        return [sequence.decode('utf-8') for sequence in sequences.numpy()]

    def compute_logits(
        self, 
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor, 
        index: tf.Tensor
    ) -> tf.Tensor:
        inputs = (
            decoder_sequence if encoder_sequence is None else 
            (decoder_sequence, encoder_sequence)
        )
        return tf.gather_nd(self.model(inputs), index)

    def _target_index(self, sequence: tf.Tensor) -> tf.Tensor:
        batch_size = keras.ops.shape(sequence)[0]
        row_index, col_index = keras.ops.where(sequence != self.pad_token_id)
        target_col_index = keras.ops.segment_max(col_index, row_index)
        target_index = keras.ops.stack([
            keras.ops.arange(batch_size, dtype=target_col_index.dtype),
            target_col_index,
        ], axis=1)
        return target_index


class TopKSampler(Sampler):
    
    def __init__(
        self,
        model: keras.models.Model, 
        tokenizer: tokenizers.Tokenizer,
        k: int = 5,
        temperature: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model, 
            tokenizer=tokenizer,
            **kwargs,
        )
        self.k = k
        self.temperature = temperature

    def __call__(
        self,
        decoder_sequence: tf.Tensor,
        encoder_sequence: tf.Tensor = None,
        index: tf.Tensor = None,
        max_iter: int = None,
    ) -> tf.Tensor:

        def cond(
            decoder_sequence: tf.Tensor, 
            _: tf.Tensor
        ) -> bool:
            all_completed = keras.ops.all(
                keras.ops.any(
                    decoder_sequence == self.eos_token_id, axis=-1
                )
            )
            return keras.ops.logical_not(all_completed)
        
        def body(
            decoder_sequence: tf.Tensor, 
            index: tf.Tensor
        ) -> tuple[tf.Tensor, tf.Tensor]:
            logits = self.compute_logits(
                decoder_sequence, encoder_sequence, index
            )
            token = self._next_token(logits)
            index += keras.ops.array([[0, 1]], dtype=index.dtype)
            decoder_sequence = keras.ops.scatter_update(
                decoder_sequence, index, token)
            return decoder_sequence, index
        
        loop_vars = (decoder_sequence, index)
        completed_decoder_sequence, _ = keras.ops.while_loop(
            cond, body, loop_vars, max_iter,
        )
        return completed_decoder_sequence

    def _sample_token(
        self,
        logits: tf.Tensor,
    ) -> tf.Tensor:
        return tf.random.categorical(
            keras.ops.log_softmax(logits), num_samples=1, dtype='int32'
        )

    def _next_token(
        self,
        logits: tf.Tensor,
    ) -> tf.Tensor:
        if self.k:
            logits, indices = keras.ops.top_k(logits, self.k, sorted=False)
        token = self._sample_token(logits / self.temperature)
        if self.k:
            token = keras.ops.take_along_axis(indices, token, axis=-1)
        return keras.ops.squeeze(token, axis=-1)
    