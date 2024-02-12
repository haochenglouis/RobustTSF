# (ICLR 2024) RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies

This code is a PyTorch implementation of our ICLR'24 paper "RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies". [[arXiv]](https://arxiv.org/abs/2402.02032)
## Prerequisites

cvxpy

Python 3.9

PyTorch 1.12.0

Torchvision 0.13.0

## Data downloading:

Download the data following the guideline from [Autoformer](https://github.com/thuml/Autoformer).

Modify the data path in dataset.py [line 27 ~ line 30].

## Running different algorithms on Electricity dataset

Please note that, in addition to the experiments provided in the RobustTSF_SOTA_Methods folder, you can totally execute the experiments using CPU.

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

## RobustTSF with SOTA architectures

In **RobustTSF_SOTA_Methods** folder, we incorporate RobustTSF with current SOTA transformer architectures.

## Citing RobustTSF

If you find this repository useful for your research, please cite it in BibTeX format:

```tex
@inproceedings{cheng2024robusttsf,
      title={RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies}, 
      author={Hao Cheng and Qingsong Wen and Yang Liu and Liang Sun},
      year={2024},
      booktitle={Proceedings of the 12th International Conference on Learning Representations},
      pages={1-28}
}
```
In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
