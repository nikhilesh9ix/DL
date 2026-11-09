# Deep Learning Lab Programs

Lab programs for Weeks 1 to 10, written in simple, plain Python.

Weeks 1, 2, 4 and 5 each have two versions:
- the main file, which is the lab record version
- a `_PDF` file, which is the exam-guide version

The two versions use different inputs and give different outputs, so learn the one your lab record uses.

| File | Program | Needs typed input? |
|---|---|---|
| `Basics_McCulloch_Pitts.py` | McCulloch-Pitts neuron (weighted sum + threshold) | Yes |
| `Basics_Perceptron.py` | Perceptron learning rule | Yes |
| `Week01_Normalization_Activation.py`, `_PDF.py` | Normalization + activation functions | Yes |
| `Week02_Logistic_Regression.py`, `_PDF.py` | Logistic regression with gradient descent | Yes |
| `Week03_Tanh_Classification.py` | Tanh + cross-entropy (observation logic) | Yes |
| `Week04_DNN_Two_Hidden_Layers.py`, `_PDF.py` | NumPy DNN, two ReLU hidden layers | Yes |
| `Week05_LR_vs_DNN.py`, `_PDF.py` | Logistic Regression vs DNN on two-moons | Main: yes / PDF: no |
| `Week06_Regularization.py` | None vs L1 vs L2 (make_moons) | No |
| `Week07_Batch_vs_MiniBatch_SGD.py` | Full-batch vs mini-batch SGD (Fashion-MNIST) | No |
| `Week08_CNN_MNIST.py` | Pooling demo, convolution kernels demo, CNN on MNIST | No |
| `Week09_ResNet_MNIST.py` | ResNet with residual blocks on MNIST | No |
| `Week10_Bidirectional_RNN_Sentiment.py` | Bidirectional SimpleRNN on IMDB | No |

## Practice inputs and expected outputs

**Basics_McCulloch_Pitts.py**
- Input: `3`, then W = `1, 1, 1`, then threshold `2`.
- Output: the output is 1 only for rows whose weighted sum is 2 or more.

**Basics_Perceptron.py**
- Input: `2`, then W = `0, 1`, then bias `0`, learning rate `1`, then targets `0, 1, 1, 1` (OR gate).
- Output: `Final weights: [1.0, 1.0]`, `Final bias: -1.0`, `Epochs: 3`, `Status: Converged`.

**Week01_Normalization_Activation.py**
- Input: `3`, then X = `2, 4, 6`, then W = `1, 2, 1`, then bias `1`, normalization `1`, activation `5` (ReLU).
- Output: `Without normalization: 17.0` and `With normalization: 3.0`.

**Week02_Logistic_Regression.py**
- Input: samples `4`, features `2`, then X = `0 0, 0 1, 1 0, 1 1`, then targets `0, 1, 1, 1`, then W = `0, 0`, bias `0`, learning rate `1`.
- Output: `Weights: [7.19755372 7.19755372]`, `Bias: -3.3595...`, `Prediction: [0 1 1 1]`.

**Week02_Logistic_Regression_PDF.py**
- Input: the same data as above, plus tolerance `0.000001`.
- Output: `Weights: [5.254036 5.254036]`, `Bias: -2.370304`, `Iterations: 1822`.

**Week03_Tanh_Classification.py**
- Input: samples `2`, features `1`, then X = `0, 1`, then targets `0, 1`, then W = `0`, bias `0`, learning rate `0.1`, tolerance `0.000001`.
- Output: `Cost: 11.51292546`, `Iterations: 2`, `Predicted outputs: [0 0]`.
- Why: training gets stuck, because tanh(0) = 0 makes the gradient 0.

**Week04_DNN_Two_Hidden_Layers.py**
- Input: samples `4`, features `2`, then X = `1 2, 2 3, 5 6, 6 7`, then targets `0, 0, 1, 1`, then hidden layers `4` and `3`, learning rate `0.1`.
- Output: `Loss: 5.2007...e-05` and `Prediction: [0 0 1 1]`.

**Week04_DNN_Two_Hidden_Layers_PDF.py**
- Input: the same data as above.
- Output: `Final loss: 0.0042703` and `Iterations: 1291`.

**Week05_LR_vs_DNN.py**
- Input: samples `1000`, noise `0.2`.
- Output: Logistic Regression gets `Accuracy: 0.855`. The DNN gets about 0.98–0.99, and this changes each run.

**Week05_LR_vs_DNN_PDF.py**
- Input: none.
- Output: Logistic Regression gets `Accuracy: 92.0 %` and `Confusion Matrix [[95 5] [11 89]]`. The DNN gets about 99%.

**Week06_Regularization.py**
- Output: all three (NONE, L1, L2) score about 94–97%.

**Week07_Batch_vs_MiniBatch_SGD.py**
- Output: Full-Batch scores low (10–35%) because it makes only 3 weight updates. Mini-Batch scores about 80–84%.

**Week08_CNN_MNIST.py**
- Output:
  - Max pooling gives `[[6, 8], [14, 16]]`.
  - Average pooling gives `[[3.5, 5.5], [11.5, 13.5]]`.
  - A 4-image convolution plot opens.
  - The CNN test accuracy is about 97–99%.

**Week09_ResNet_MNIST.py**
- Output: test accuracy is about 96–98.5%, then a 2×5 grid of digits opens, each titled `True:x Pred:y`.

**Week10_Bidirectional_RNN_Sentiment.py**
- Output: test accuracy is about 60–75%, then 10 lines like `Review 1: Predicted=..., Actual=...`.

## Requirements

```
pip install numpy scikit-learn tensorflow matplotlib
```
