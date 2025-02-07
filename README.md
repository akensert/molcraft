<img src="https://github.com/akensert/molcraft/blob/main/media/molcraft-logo.png" alt="molcraft-logo" width="90%">

**Transformers** with **TensorFlow** and **Keras**. Focused on **Molecule Generation** and **Chemistry Predictions**.

> [!NOTE]
> In progress.

## Highlights

Aims to implement efficient models, samplers and \[soon\] reinforcement learning for [SMILES](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System) generation and optimization.

- [Models](https://github.com/akensert/molcraft/blob/main/molcraft/models.py) / [Layers](https://github.com/akensert/molcraft/blob/main/molcraft/layers.py)
    - Implements **key-value caching** for efficient autoregression
- [Samplers](https://github.com/akensert/molcraft/blob/main/molcraft/samplers.py)
    - Samples [Models](https://github.com/akensert/molcraft/blob/main/molcraft/models.py) for next tokens
    - Can **generate** a batch of **sequences** in parallel **non-eagerly**
    - Can generate a batch of sequences based on **initial sequences of varying lengths**
- [Tokenizers](https://github.com/akensert/molcraft/blob/main/molcraft/tokenizers.py)
    - Tokenizes data input for [Models](https://github.com/akensert/molcraft/blob/main/molcraft/models.py)
    - Can be **adapted** to data via **tokenizer.adapt(ds)** to build vocabulary
    - Can be added as a layer to **keras.Sequential**
    - Can both **tokenize** and **detokenize** data 

## Code Examples

```python
import tensorflow as tf
import keras
import random

from molcraft import tokenizers
from molcraft import models
from molcraft import samplers 

filename = './data/zinc250K.txt' # replace this with actual path

with open(filename, 'r') as fh:
    smiles = fh.read().splitlines()

random.shuffle(smiles)

# Adapt tokenizer (create vocabulary)
tokenizer = tokenizers.SMILESTokenizer(add_bos=True, add_eos=True)
tokenizer.adapt(smiles)

# Build dataset (input pipeline)
ds = tf.data.Dataset.from_tensor_slices(smiles)
ds = ds.shuffle(8192)
ds = ds.batch(256)
ds = ds.map(tokenizer)
ds = ds.map(lambda x: (x[:, :-1], x[:, 1:]))
ds = ds.prefetch(-1)

# Build, compile, and fit model
model = models.TransformerDecoder(
    num_layers=4,
    num_heads=8,
    embedding_dim=512,
    intermediate_dim=1024,
    vocabulary_size=tokenizer.vocabulary_size,
    sequence_length=tokenizer.sequence_length,
    dropout=0,
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4), 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
model.fit(ds, epochs=1)

# Generate 32 novel SMILES with sampler
sampler = samplers.TopKSampler(model, tokenizer)
smiles = sampler.sample([''] * 32)
```

## Installation
> [!NOTE]
> Project is under development, hence incomplete and subject to breaking changes.

For GPU users:
```
git clone git@github.com:akensert/molcraft.git
pip install -e .[gpu]
```
For CPU users:
```
git clone git@github.com:akensert/molcraft.git
pip install -e .
```
