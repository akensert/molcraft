import tensorflow as tf
import tensorflow_text as tf_text

import keras
from keras import ops

import re

import numpy as np

import warnings

from molcraft.definitions import SpecialTokens
from molcraft.definitions import SMILES_REGEX_PATTERN


warnings.simplefilter('always', UserWarning)


class SMILESTokenizer(keras.layers.Layer):
    
    def __init__(
        self, 
        vocab: list[str] = None,
        max_sequence_length: int = None,
        id_dtype: tf.dtypes.DType = tf.int32,
        adapt_eagerly: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if vocab is not None and not tf.is_tensor(vocab):
            vocab = tf.convert_to_tensor(vocab)
        self._vocab = vocab
        self._max_sequence_length = max_sequence_length
        self._id_dtype = id_dtype
        self._key_dtype = tf.string
        self._value_dtype = self._id_dtype
        self._smiles_regex_pattern = SMILES_REGEX_PATTERN
        self._mask_token = SpecialTokens.MASK
        self._bos_token = SpecialTokens.BOS
        self._eos_token = SpecialTokens.EOS
        self._unk_token = SpecialTokens.UNK
        self._mask_token_id = 0
        self._unk_token_id = 1
        self._bos_token_id = 2
        self._eos_token_id = 3
        self._unk_token_regex = re.escape(self._unk_token)
        self._pattern_remove = '|'.join([
            re.escape(self._mask_token), 
            re.escape(self._bos_token), 
            re.escape(self._eos_token),
        ])
        self._vocabulary_size = None 
        self._adapt_eagerly = adapt_eagerly
        if not self._adapt_eagerly:
            self._adapt_fn = tf.function(self._adapt_fn)

        self._build()

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        with open(filename) as fh:
            vocab = fh.read().splitlines()
        return cls(vocab, **kwargs)

    def tokenize(self, smiles: tf.Tensor) -> tf.Tensor:
        smiles = self._prepare_smiles(smiles)
        tokens = self._tokenize_smiles(smiles)
        return self._vectorize_tokens(tokens)
    
    def detokenize(self, tokens: tf.Tensor) -> tf.Tensor:
        tokens = self._lookup_table_reverse.lookup(tokens)
        return self._join_tokens(tokens)

    def call(self, inputs):
        return self.tokenize(inputs)
    
    def adapt(self, data, batch_size: int = 4096):
        
        self._adapt_table = tf.lookup.experimental.MutableHashTable(
            key_dtype=self._key_dtype, 
            value_dtype=self._value_dtype, 
            default_value=0
        )
        
        ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)

        total = len(data)
        
        progbar = keras.utils.Progbar(target=total)
        
        global_max_seq_length = 0
        steps = total // batch_size 
        remainder = total % batch_size
        n = 0
        for i, smiles in ds.enumerate():
            max_seq_length = self._adapt_fn(smiles).numpy()
            n += batch_size if i < steps else remainder
            progbar.update(n)
            if max_seq_length > global_max_seq_length:
                global_max_seq_length = max_seq_length
        
        self._max_sequence_length = global_max_seq_length
        self._construct_lookup_tables()

    @property
    def vocabulary_size(self):
        return self._vocabulary_size
    
    @property
    def vocabulary(self) -> list[str]:
        keys, values = self._lookup_table.export()
        index = np.argsort(values.numpy())
        return keys.numpy()[index]
    
    @property
    def max_sequence_length(self):
        return self._max_sequence_length
    
    def _adapt_fn(self, x: tf.Tensor) -> int:
        x = self._tokenize_smiles(x)
        max_seq_length = x.bounding_shape(axis=1) + 2
        if isinstance(x, tf.RaggedTensor):
            x = x.flat_values
        tokens, _, counts = tf.unique_with_counts(x, out_idx=self._value_dtype)
        self._adapt_table.insert(
            tokens, counts + self._adapt_table.lookup(tokens))
        return max_seq_length
    
    def _build(self):
        if self._vocab is None:
            self._built = False
            self._lookup_table = None
            self._lookup_table_reverse = None
            return None 
        self._construct_lookup_tables()
        
    def _prepare_smiles(self, smiles: tf.Tensor) -> tf.Tensor:
        batch_size = ops.shape(smiles)[0]
        return tf.strings.reduce_join(
            tf.stack([
                tf.repeat([self._bos_token], [batch_size]), 
                smiles, 
                tf.repeat([self._eos_token], [batch_size])
            ]), 
            axis=0
        )

    def _tokenize_smiles(self, smiles: tf.Tensor) -> tf.RaggedTensor:
        return tf_text.regex_split(
            smiles, 
            delim_regex_pattern=self._smiles_regex_pattern, 
            keep_delim_regex_pattern=self._smiles_regex_pattern
        )

    def _vectorize_tokens(self, tokens: tf.RaggedTensor) -> tf.Tensor:
        return self._lookup_table.lookup(tokens).to_tensor(
            shape=[None, self._max_sequence_length]
        )
    
    def _join_tokens(self, tokens: tf.Tensor) -> tf.Tensor:
        smiles = tf.strings.reduce_join(tokens, axis=-1)
        if tf.reduce_any(
            tf.strings.regex_full_match(
                smiles, '.*' + self._unk_token_regex + '.*'
            )
        ):
            warnings.warn(
                f'Unknown token {self._unk_token!r} found in decoded SMILES.'
            )
        return tf.strings.regex_replace(
            smiles, self._pattern_remove, '')
    
    def _construct_lookup_tables(self) -> None:
        
        if self._vocab is not None:
            tokens = self._vocab
        else:
            tokens, counts = self._export_adapted_table()
            sorted_indices = np.lexsort((tokens.numpy(), counts.numpy()))[::-1]
            tokens = tf.gather(tokens, sorted_indices)
            self._vocab = tokens

        tokens = tf.concat([
            [self._mask_token], 
            [self._unk_token],
            [self._bos_token], 
            [self._eos_token], 
            tokens,
        ], axis=0)
        
        self._vocabulary_size = len(tokens)

        ids = tf.range(tf.size(tokens), dtype=self._value_dtype)
                           
        self._lookup_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tokens, 
                values=ids, 
                key_dtype=self._key_dtype, 
                value_dtype=self._value_dtype
            ), 
            1
        )
        self._lookup_table_reverse = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=ids, 
                values=tokens, 
                key_dtype=self._value_dtype, 
                value_dtype=self._key_dtype
            ), 
            self._unk_token
        )

        self._built = True

    def _export_adapted_table(self) -> tuple[tf.Tensor, tf.Tensor]:
        self._adapt_table.remove(
            tf.convert_to_tensor([
                self._mask_token, 
                self._bos_token, 
                self._eos_token, 
                self._unk_token, 
            ])
        )
        tokens, counts = self._adapt_table.export()
        self._adapt_table.remove(tokens)
        return tokens, counts 
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'vocab': list(self._vocab.numpy().astype(str)),
            'max_sequence_length': self._max_sequence_length,
            'id_dtype': self._id_dtype,
            'adapt_eagerly': self._adapt_eagerly,
        })
        return config
    
    def compute_output_shape(
        self, 
        input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape)
        return input_shape.concatenate(
            tf.TensorShape([self._max_sequence_length]))
    
    def compute_output_signature(
        self, 
        inputs: tf.TensorSpec
    ) -> tf.TensorSpec:
        input_shape = inputs.shape 
        input_dtype = inputs.dtype
        return tf.TensorSpec(
            input_shape.concatenate(tf.TensorShape([self._max_sequence_length])), 
            input_dtype)