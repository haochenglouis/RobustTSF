# RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies

This code is a PyTorch implementation of our paper "RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies".
## Prerequisites

cvxpy

Python 3.9

PyTorch 1.12.0

Torchvision 0.13.0

## Data downloading:

Download the data following the guideline from [Autoformer](https://github.com/thuml/Autoformer).

Modify the data path in dataset.py [line 27 ~ line 30].

## Running different algorithms on Electricity dataset

### Vanilla training (MAE) with Missing anomaly (anomaly ratio 0.3): 

```
python train_noisy.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

ano_type: {const, missing, gaussian}

dataset: {ele, traffic}

loss: {mae, mse}

### Offline inputation with Missing anomaly (anomaly ratio 0.3): 

First get a pre-trained model: 

```
python train_noisy.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

Then using the pretrained model to perform offline detection-imputation-retraining

```
python train_noisy.py --impute model_impute --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

### Onine inputation with Missing anomaly (anomaly ratio 0.3): 

First get a pre-trained model: 

```
python train_noisy.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

Then using the pretrained model to perform online detection-imputation-retraining

```
python train_noisy_online.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

### Loss-based selection with Missing anomaly (anomaly ratio 0.3): 

```
python train_noisy_mv_selection.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```

### RobustTSF (ours) with Missing anomaly (anomaly ratio 0.3): 

```
python train_noisy_weighting.py --loss mae --dataset ele --ano_type missing --ano_ratio 0.3 
```
