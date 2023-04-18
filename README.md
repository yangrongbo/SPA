## Preparation

1. Downloading datasets from [here](https://drive.google.com/drive/folders/1b4GmVfnYhV7BmGM21E7harBJ2fqCuM6z?usp=sharing) .
2. Download the models from [here](https://drive.google.com/drive/folders/1YYpdg1uApinKrAqw7XF1XiQKRhG5GFi_?usp=sharing) .
3. You can also download [NRP](https://drive.google.com/drive/folders/1F43MNFqJ6I5ph4z_gxZignxcTWx4HXfg?usp=sharing) and [RS](https://drive.google.com/drive/folders/1iZIDHnfCEVu_pEM8CO3XQtdrUklGzddx?usp=sharing) defense methods for attacks.

## Implementation

- Run commands directly such as:


```python
python SPA_MI_COZ.py
```

- The produced adversarial examples are saved to the specified folder and then run the following command to evaluate them:


```python
python evaluate.py
```

- The following instructions can be used to attack six defense methods: R&P, Bit Red, JPEG, FD, RS, and NRP.

```
python evaluate_FD.py
```
