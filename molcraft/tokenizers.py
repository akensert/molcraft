import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import keras
import re
import abc 


PAD_TOKEN = '[pad]'
PAD_TOKEN_ID = 0
OOV_TOKEN = '[oov]'
OOV_TOKEN_ID = 1
BOS_TOKEN = '[bos]'
EOS_TOKEN = '[eos]'
SEP_TOKEN = '[sep]'

TOKEN_DTYPE = tf.string
TOKEN_ID_DTYPE = tf.int32
TOKEN_COUNT_DTYPE = tf.int64

DEFAULT_ADAPT_BATCH_SIZE = 8192


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
        padding_mode: str = 'right',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if sequence_length and not isinstance(sequence_length, int):
            raise ValueError('`sequence_length` needs to be an int or None.')
        self.sequence_length = sequence_length
        self.padding_mode = 'left' if padding_mode == 'left' else 'right'
        self.pad_token = PAD_TOKEN
        self.bos_token = BOS_TOKEN if add_bos else None
        self.eos_token = EOS_TOKEN if add_eos else None
        self.oov_token = oov_token
        self.sep_token = sep_token
        self.pad_token_id = PAD_TOKEN_ID
        self.oov_token_id = OOV_TOKEN_ID
        self.bos_token_id = 2 if self.bos_token else None 
        self.eos_token_id = 3 if self.eos_token else None 
        self.eos_token_id -= 1 if not self.bos_token else 0 
        self._special_tokens = [self.pad_token]
        if self.oov_token:
            self._special_tokens.append(self.oov_token)
        if self.bos_token:
            self._special_tokens.append(self.bos_token)
        if self.eos_token:
            self._special_tokens.append(self.eos_token)
        
        if vocabulary:
            adapted_table = tf.lookup.experimental.MutableHashTable(
                key_dtype=TOKEN_DTYPE,
                value_dtype=TOKEN_COUNT_DTYPE,
                default_value=0
            )
            adapted_table.insert(
                vocabulary, 
                keras.ops.arange(
                    len(vocabulary), 0, -1, dtype=TOKEN_COUNT_DTYPE
                )
            )
            self._create_vocab(adapted_table)
            self._create_lookup_tables()
    
    @abc.abstractmethod
    def pretokenize(self, inputs: tf.Tensor) -> tf.RaggedTensor:
        '''Pretokenizes sequence(s).
        
        Namely, this method splits sequence(s) into parts (tokens).

        Used by the `adapt` method to build vocabulary, and called by 
        `tokenize` to generate token ids.

        Should accept a batch of sequences.
        '''

    @abc.abstractmethod
    def detokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        '''Detokenizes tokenized sequence(s).

        Should reverse `tokenize` to obtain initial sequence(s).

        Should accept a batch of sequences.
        '''

    def tokenize(
        self, 
        inputs: tf.Tensor, 
        ragged: bool = False,
    ) -> tf.RaggedTensor | tf.Tensor:
        '''Tokenizes sequence(s).

        While `pretokenize` split sequence(s) into tokens, `tokenize` splits 
        sequence(s) into token indices. 
        
        `tokenize` should implement `pretokenize`.

        Should accept a batch of sequences and should incorporate `pretokenize`. 
        '''
        inputs = tf.strings.join([
            self.bos_token or '', inputs, self.eos_token or ''])
        tokens = self.pretokenize(inputs)
        token_ids = self.token_to_id(tokens)
        if not self.sequence_length or ragged:
            return token_ids 
        if self.padding_mode == 'left':
            token_ids = token_ids[:, ::-1]
        token_ids = token_ids.to_tensor(
            shape=(None, self.sequence_length), 
            default_value=self.pad_token_id,
        )
        if self.padding_mode == 'left':
            token_ids = token_ids[:, ::-1]
        return token_ids
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        '''Calls the `Tokenizer` layer.
        '''
        return self.tokenize(inputs)
    
    def adapt(
        self, 
        data: list[str] | tf.data.Dataset,
        adapt_sequence_length: bool = True,
    ) -> None:
        '''Adapts the `Tokenizer`.

        If a vocabulary is not supplied with the `Tokenizer` instance, it can 
        construct a vocabulary based on a data set of sequences.
        '''

        adapted_table = tf.lookup.experimental.MutableHashTable(
            key_dtype=TOKEN_DTYPE,
            value_dtype=TOKEN_COUNT_DTYPE,
            default_value=0
        )

        def adapt_fn(inputs: tf.Tensor) -> int:
            tokens = self.pretokenize(inputs)
            if isinstance(tokens, tf.RaggedTensor):
                max_sequence_length = tokens.bounding_shape(axis=1)
                tokens = tokens.flat_values
            else:
                max_sequence_length = keras.ops.shape(tokens)[1]
            tokens, _, counts = tf.unique_with_counts(
                tokens, out_idx=TOKEN_COUNT_DTYPE
            )
            adapted_table.insert(tokens, counts + adapted_table.lookup(tokens))
            return max_sequence_length

        if not isinstance(data, tf.data.Dataset):
            ds = tf.data.Dataset.from_tensor_slices(data)
            ds = ds.batch(DEFAULT_ADAPT_BATCH_SIZE)
            ds = ds.prefetch(-1)

        global_max_sequence_length = 0
        for x in ds:
            max_sequence_length = adapt_fn(x)
            global_max_sequence_length = max(
                global_max_sequence_length, max_sequence_length
            )

        if adapt_sequence_length:
            self.sequence_length = int(global_max_sequence_length)
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
        self.built = True

    def token_to_id(self, token, /) -> tf.RaggedTensor | tf.Tensor:
        if not hasattr(self, '_lookup_table'):
            raise ValueError(
                'Lookup tables have not yet been constructed. '
                'Either adapt the tokenizer to data (via the `adapt` method) '
                'or pass a vocabulary to its constructor.'
            )
        if not keras.ops.is_tensor(token):
            token = keras.ops.convert_to_tensor(token)
        return self._lookup_table.lookup(token)

    def id_to_token(self, index, /) -> tf.RaggedTensor | tf.Tensor:
        if not hasattr(self, '_lookup_table_reverse'):
            raise ValueError(
                'Lookup tables have not yet been constructed. '
                'Either adapt the tokenizer to data (via the `adapt` method) '
                'or pass a vocabulary to its constructor.'
            )
        if not keras.ops.is_tensor(index):
            index = tf.convert_to_tensor(index)
        return self._lookup_table_reverse.lookup(index)

    def pad(
        self, 
        inputs: tf.RaggedTensor, 
        sequence_length: int | str | None = 'auto',
        pad_value: int | str | None = 'auto'
    ) -> tf.Tensor:
        sequence_length = (
            self.sequence_length if sequence_length == 'auto' else
            sequence_length
        )
        pad_value = (
            self.pad_token_id if pad_value == 'auto' else pad_value
        )
        return inputs.to_tensor(
            shape=(None, sequence_length), default_value=pad_value
        )
    
    def get_vocabulary(self):
        return getattr(self, '_vocabulary', None)

    @property
    def vocabulary_size(self):
        return getattr(self, '_vocabulary_size', None)

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
            tf.TensorShape([self.sequence_length]))
    
    def compute_output_signature(
        self, 
        inputs: tf.TensorSpec
    ) -> tf.TensorSpec:
        input_shape = inputs.shape 
        input_dtype = inputs.dtype
        return tf.TensorSpec(
            input_shape.concatenate(tf.TensorShape([self.sequence_length])), 
            input_dtype)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'vocabulary': self.get_vocabulary(),
            'sequence_length': self.sequence_length,
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

    REGEX_PATTERN_CLEANUP = (
        re.escape(PAD_TOKEN) + '|' + 
        re.escape(BOS_TOKEN) + '|' + 
        re.escape(EOS_TOKEN) + '.*'
    )
    
    def pretokenize(self, inputs: tf.Tensor) -> tf.RaggedTensor:
        return tf_text.regex_split(
            inputs, 
            delim_regex_pattern=self.SMILES_REGEX_PATTERN, 
            keep_delim_regex_pattern=self.SMILES_REGEX_PATTERN
        )
    
    def detokenize(self, inputs: tf.Tensor) -> tf.Tensor:
        tokens = self.id_to_token(inputs)
        sequences = tf.strings.reduce_join(tokens, axis=-1)
        sequences = tf.strings.regex_replace(
            sequences, self.REGEX_PATTERN_CLEANUP, ''
        )
        return sequences

    @staticmethod
    def truncate_product(smiles: str, sep: str = SEP_TOKEN):
        if isinstance(smiles, str):
            return sep.join(smiles.split(sep)[:-1]) + sep 
        array_type = smiles.__class__()
        return array_type(
            list(map(lambda s: sep.join(s.split(sep)[:-1]) + sep, smiles))
        )

    @staticmethod
    def extract_product(smiles: str, sep: str = SEP_TOKEN):
        if isinstance(smiles, str):
            return smiles.split(sep)[-1]
        array_type = smiles.__class__()
        return array_type(
            list(map(lambda s: s.split(sep)[-1]), smiles)
        )