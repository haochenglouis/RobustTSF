# (ICLR 2024) RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies

This code is a PyTorch implementation of our ICLR'24 paper "RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies". [[arXiv]](https://arxiv.org/abs/2402.02032)
## Citing RobustTSF
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

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




## Further Reading
1, [**Transformers in Time Series: A Survey**](https://arxiv.org/abs/2202.07125), in IJCAI 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)

**Authors**: Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun

```bibtex
@inproceedings{wen2023transformers,
  title={Transformers in time series: A survey},
  author={Wen, Qingsong and Zhou, Tian and Zhang, Chaoli and Chen, Weiqi and Ma, Ziqing and Yan, Junchi and Sun, Liang},
  booktitle={International Joint Conference on Artificial Intelligence(IJCAI)},
  year={2023}
}
```

2, [**Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook**](https://arxiv.org/abs/2310.10196), in *arXiv* 2023.
[\[GitHub Repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

**Authors**: Ming Jin, Qingsong Wen*, Yuxuan Liang, Chaoli Zhang, Siqiao Xue, Xue Wang, James Zhang, Yi Wang, Haifeng Chen, Xiaoli Li (IEEE Fellow), Shirui Pan*, Vincent S. Tseng (IEEE Fellow), Yu Zheng (IEEE Fellow), Lei Chen (IEEE Fellow), Hui Xiong (IEEE Fellow)

```bibtex
@article{jin2023lm4ts,
  title={Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook}, 
  author={Ming Jin and Qingsong Wen and Yuxuan Liang and Chaoli Zhang and Siqiao Xue and Xue Wang and James Zhang and Yi Wang and Haifeng Chen and Xiaoli Li and Shirui Pan and Vincent S. Tseng and Yu Zheng and Lei Chen and Hui Xiong},
  journal={arXiv preprint arXiv:2310.10196},
  year={2023}
}
```

3, [**Position Paper: What Can Large Language Models Tell Us about Time Series Analysis**](https://arxiv.org/abs/2402.02713), in *arXiv* 2024.

**Authors**: Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang*, Bin Yang, Jindong Wang, Shirui Pan, Qingsong Wen*


```bibtex
@article{jin2024position,
   title={Position Paper: What Can Large Language Models Tell Us about Time Series Analysis}, 
   author={Ming Jin and Yifan Zhang and Wei Chen and Kexin Zhang and Yuxuan Liang and Bin Yang and Jindong Wang and Shirui Pan and Qingsong Wen},
  journal={arXiv preprint arXiv:2402.02713},
  year={2024}
}
```
4, [**AI for Time Series (AI4TS) Papers, Tutorials, and Surveys**](https://github.com/qingsongedu/awesome-AI-for-time-series-papers)
 
