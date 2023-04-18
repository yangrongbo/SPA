## Preparation

1. Downloading datasets from [here](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) to the dataset folder.

2. Download the models from [here](https://github.com/ylhz/tf_to_pytorch_model) to the models folder.

## Implementation

Run commands directly such as:

```python
python SPA_MI_FGSM.py
```

The produced adversarial samples are then saved to the specified folder and we then run the following command to evaluate them:

```python
python evaluate.py
```


We have implemented R&P, Bit-Red, JPEG and FD, RS and NRP we provide references as follows：

+ RS: https://github.com/locuslab/smoothing. 
+ NRP: https://github.com/Muzammal-Naseer/NRP. 

After creating the adversarial examples, you can run the following command directly:

```python
python evaluate_RP.py
```
