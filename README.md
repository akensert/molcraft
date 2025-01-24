<img src="https://github.com/akensert/molcraft/blob/main/media/molcraft-logo.png" alt="molcraft-logo" width="90%">

**Transformers** with **TensorFlow** and **Keras**. Focused on molecule generation and chemistry prediction.

## Highlights

- [Tokenizers](https://github.com/akensert/molcraft/blob/main/molcraft/tokenizers.py)
    - Can be **adapted** to data via **tokenizer.adapt(ds)**
    - Can be added as a layer to **keras.Sequential**
    - Can both **tokenize** and **detokenize** data 
- [Samplers](https://github.com/akensert/molcraft/blob/main/molcraft/samplers.py)
    - Samples model for next token
    - Can **sample** tokens **non-eagerly**
    - Can **sample** tokens for a **batch of data**, containing sequences of varying lengths

## Installation
> [!NOTE]
> Project is under development, hence subject to breaking changes.

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
