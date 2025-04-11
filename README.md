<img src="https://github.com/akensert/molcraft/blob/main/docs/_static/molcraft-logo.png" alt="molcraft-logo">

**Deep Learning on Molecules**: A Minimalistic GNN package for Molecular ML. 

> [!NOTE]  
> In progress/Unfinished.

## Highlights
- Compatible with **Keras 3**
- Simplified API
- Fast featurization
- Modular graph **layers**
- Serializable graph **featurizers** and **models**
- Flexible **GraphTensor**

## Examples 

```python
from molcraft import features
from molcraft import descriptors
from molcraft import featurizers 
from molcraft import layers
from molcraft import models 
import keras

featurizer = featurizers.MolGraphFeaturizer(
    atom_features=[
        features.AtomType(),
        features.TotalNumHs(),
        features.Degree(),
    ],
    bond_features=[
        features.BondType(),
        features.IsRotatable(),
    ],
    super_atom=True,
    self_loops=False,
)

graph = featurizer([('N[C@@H](C)C(=O)O', 2.0), ('N[C@@H](CS)C(=O)O', 1.0)])
print(graph)

model = models.create(
    layers.Input(graph.spec),
    layers.NodeEmbedding(dim=128),
    layers.EdgeEmbedding(dim=128),
    layers.GraphTransformer(units=128),
    layers.GraphTransformer(units=128),
    layers.GraphTransformer(units=128),
    layers.GraphTransformer(units=128),
    layers.Readout(mode='mean'),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dense(1)
)

pred = model(graph)
print(pred)
```

## Installation

Install the pre-release of molcraft via pip:

```bash
pip install molcraft --pre
```

with GPU support:

```bash
pip install molcraft[gpu] --pre
```