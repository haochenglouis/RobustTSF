# RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies

This code showcases the capability of RobustTSF in enhancing State-of-the-Art (SOTA) methods.

Within **Exp_Transfomer**, you will find prominent methods such as AutoFormer, Informer, and Fedformer. In contrast, **Robust_Exp_Transfomer** seamlessly incorporates RobustTSF into AutoFormer, Informer, and Fedformer.

## Running Guidelines

To execute the code successfully, navigate to each folder and run the scripts located under **./scripts**. Note you need to first modify the data path in run.py

To ensure complete consistency, we replicated all experimental settings from the FedFormer official code, encompassing input length (96), train/validation/test split, evaluation protocol, training epochs, and three independent runs for each setting. Notably, we solely incorporated RobustTSF into each SOTA architecture, emphasizing its model-agnostic nature. We present the results below, reporting the Mean Absolute Error (MAE) from the best epoch for each setting (averaged over three independent runs):

|   Dataset: Electricity    | Const 0.1 | Const 0.3 | Missing 0.1 | Missing 0.3 | Gaussian 0.1 | Gaussian 0.3 |
| :-----------------------: | :-------: | :-------: | :---------: | :---------: | :----------: | :----------: |
|        Autoformer         |   0.279   |   0.281   |    0.309    |    0.341    |    0.315     |    0.396     |
| RobustTSF with Autoformer | **0.237** | **0.252** |  **0.243**  |  **0.289**  |  **0.271**   |  **0.316**   |
|         Informer          |   0.169   |   0.190   |    0.180    |    0.344    |    0.200     |    0.231     |
|  RobustTSF with Informer  | **0.156** | **0.172** |  **0.157**  |  **0.186**  |  **0.165**   |  **0.186**   |
|         Fedformer         |   0.194   |   0.203   |    0.234    |    0.271    |    0.250     |    0.312     |
| RobustTSF with Fedformer  | **0.174** | **0.179** |  **0.180**  |  **0.195**  |  **0.182**   |  **0.190**   |

Encouragingly, RobustTSF consistently enhances the performance of SOTA methods in these noisy settings.

