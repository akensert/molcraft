import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import keras
import re
import abc 


PAD_TOKEN = '[pad]'
EOS_TOKEN = '[eos]'
BOS_TOKEN = '[bos]'
OOV_TOKEN = '[oov]'

REGEX_PATTERN_OOV_TOKEN = '.*' + re.escape(OOV_TOKEN) + '.*'
REGEX_PATTERN_TRUNCATE = re.escape(EOS_TOKEN) + '.*'
REGEX_PATTERN_CLEANUP = re.escape(PAD_TOKEN) + '|' + re.escape(BOS_TOKEN)

TOKEN_DTYPE = tf.string
TOKEN_ID_DTYPE = tf.int32


@keras.saving.register_keras_serializable(package='molcraft')
class Tokenizer(keras.layers.Layer, abc.ABC):

    def __init__(
        self,
        vocabulary: list[str] = None,
        sequence_length: int = None,
        add_bos: bool = False,
        add_eos: bool = False, 
        oov_token: str = OOV_TOKEN,
        sep_token: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._sequence_length = sequence_length
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN if add_bos else None
        self.eos_token = EOS_TOKEN if add_eos else None
        self.oov_token = oov_token
        self.oov_token_id = 1
        self.sep_token = sep_token
        self._special_tokens = [self.pad_token]
        if self.oov_token:
            self._special_tokens.append(self.oov_token)
        if self.bos_token:
            self._special_tokens.append(self.bos_token)
        if self.eos_token:
            self._special_tokens.append(self.eos_token)
        if self._sequence_length:
            if not isinstance(self._sequence_length, int):
                raise ValueError('sequence_length needs to be an int or None.')
            
        if vocabulary:
            adapted_table = tf.lookup.experimental.MutableHashTable(
                key_dtype=TOKEN_DTYPE,
                value_dtype=tf.int64,
                default_value=0
            )
            adapted_table.insert(
                vocabulary, 
                keras.ops.arange(len(vocabulary), 0, -1, dtype=tf.int64)
            )
            self._create_vocab(adapted_table)
            self._create_lookup_tables()

    
    @abc.abstractmethod
    def pretokenize(self, inputs: tf.Tensor) -> tf.RaggedTensor:
        pass

    @abc.abstractmethod
    def tokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def detokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.tokenize(inputs)
    
    def adapt(self, data: list[str], batch_size: int = 4096):

        adapted_table = tf.lookup.experimental.MutableHashTable(
            key_dtype=TOKEN_DTYPE,
            value_dtype=TOKEN_ID_DTYPE,
            default_value=0
        )

        def adapt_fn(inputs: tf.Tensor) -> int:
            tokens = self.pretokenize(inputs)
            if isinstance(tokens, tf.RaggedTensor):
                tokens = tokens.flat_values
            tokens, _, counts = tf.unique_with_counts(
                tokens, out_idx=TOKEN_ID_DTYPE
            )
            adapted_table.insert(tokens, counts + adapted_table.lookup(tokens))
        
        ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
        total = len(data)
        total_steps = total // batch_size
        remainder = total % batch_size 
        progbar = keras.utils.Progbar(target=total)
        counter = 0
        for step, batch in ds.enumerate():
            adapt_fn(batch)
            counter += batch_size if step < total_steps else remainder 
            progbar.update(counter)
        
        self._create_vocab(adapted_table)
        self._create_lookup_tables()

    def _create_vocab(
        self, 
        adapted_table: tf.lookup.experimental.MutableHashTable
    ) -> None:
        adapted_table.remove(keras.ops.convert_to_tensor(self.special_tokens))
        vocab, counts = adapted_table.export()
        adapted_table.remove(vocab)
        sorted_indices = np.lexsort((vocab.numpy(), counts.numpy()))[::-1]
        vocab = tf.gather(vocab, sorted_indices)
        vocab = [token.numpy().decode('utf-8') for token in vocab]
        self._vocabulary = self.special_tokens + list(vocab)
        self._vocabulary_size = len(self._vocabulary)
    
    def _create_lookup_tables(self) -> None:
        tokens = keras.ops.convert_to_tensor(self._vocabulary)
        token_ids = keras.ops.arange(self._vocabulary_size)
        self._lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tokens, 
                values=token_ids, 
                key_dtype=TOKEN_DTYPE, 
                value_dtype=TOKEN_ID_DTYPE,
            ), 
            self.oov_token_id
        )
        self._lookup_table_reverse = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=token_ids, 
                values=tokens, 
                key_dtype=TOKEN_ID_DTYPE, 
                value_dtype=TOKEN_DTYPE,
            ), 
            self.oov_token
        )

    def token_to_id(self, token, /):
        if not hasattr(self, '_lookup_table'):
            raise ValueError(
                'Lookup tables have not yet been constructed. '
                'Either adapt the tokenizer to data (via the `adapt` method) '
                'or pass a vocabulary to its constructor.'
            )
        if not tf.is_tensor(token):
            token = tf.convert_to_tensor(token)
        return self._lookup_table.lookup(token)

    def id_to_token(self, index, /):
        if not hasattr(self, '_lookup_table_reverse'):
            raise ValueError(
                'Lookup tables have not yet been constructed. '
                'Either adapt the tokenizer to data (via the `adapt` method) '
                'or pass a vocabulary to its constructor.'
            )
        if not tf.is_tensor(index):
            index = tf.convert_to_tensor(index)
        return self._lookup_table_reverse.lookup(index)

    def get_vocabulary(self):
        return getattr(self, '_vocabulary', None)

    @property
    def vocabulary_size(self):
        return getattr(self, '_vocabulary_size', None)
    
    @property 
    def sequence_length(self):
        return self._sequence_length
    
    @property 
    def special_tokens(self):
        return self._special_tokens
    
    @property 
    def special_token_ids(self):
        return list(range(len(self._special_tokens)))
    
    def compute_output_shape(
        self, 
        input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape)
        return input_shape.concatenate(
            tf.TensorShape([self._sequence_length]))
    
    def compute_output_signature(
        self, 
        inputs: tf.TensorSpec
    ) -> tf.TensorSpec:
        input_shape = inputs.shape 
        input_dtype = inputs.dtype
        return tf.TensorSpec(
            input_shape.concatenate(tf.TensorShape([self._sequence_length])), 
            input_dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary': self.get_vocabulary(),
            'sequence_length': self._sequence_length,
            'add_bos': (self.bos_token is not None),
            'add_eos': (self.eos_token is not None),
            'oov_token': self.oov_token,
            'sep_token': self.sep_token,
        })
        return config


@keras.saving.register_keras_serializable(package='molcraft')
class SMILESTokenizer(Tokenizer):

    SMILES_REGEX_PATTERN = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|="
        r"|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    )
    
    def pretokenize(self, inputs: tf.Tensor) -> tf.RaggedTensor:
        sequences = tf.strings.join([
            self.bos_token or '', inputs, self.eos_token or ''])
        return tf_text.regex_split(
            sequences, 
            delim_regex_pattern=self.SMILES_REGEX_PATTERN, 
            keep_delim_regex_pattern=self.SMILES_REGEX_PATTERN
        )
    
    def tokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        tokens = self.pretokenize(inputs)
        token_ids = self.token_to_id(tokens)
        if isinstance(token_ids, tf.RaggedTensor):
            return token_ids.to_tensor(
                shape=(None, self.sequence_length), default_value=0)
        if self.sequence_length:
            return token_ids[:, :self.sequence_length]
        return token_ids

    def detokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        tokens = self.id_to_token(inputs)
        sequences = tf.strings.reduce_join(tokens, axis=-1)
        sequences = tf.strings.regex_replace(
            sequences, REGEX_PATTERN_TRUNCATE, ''
        )
        sequences = tf.strings.regex_replace(
            sequences, REGEX_PATTERN_CLEANUP, ''
        )
        return sequences

