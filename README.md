# Trained-Rank-Pruning
PyTorch code for "Trained Rank Pruning for Efficient Neural Networks" <https://arxiv.org/abs/1812.02402><br>
Our code is built based on  [bearpaw](https://github.com/bearpaw/pytorch-classification)<br>
<img src=framework.png width=50%><br>
What's in this repo so far:
 * TRP code for CIFAR-10 experiments
 * Nuclear regularization code for CIFAR-10 experiments
 
#### Simple Examples
Training ResNet-20 baseline:

```
python cifar-nuclear-regularization.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 

```
Training ResNet-20 with nuclear norm:

```
python cifar-nuclear-regularization.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 --nuclear-weight 0.0003

```
Training ResNet-20 with TRP:
```
python cifar-TRP.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 --nuclear-weight 0.0003

```
Decompose the trained model without retraining:
```
python cifar-nuclear-regularization.py.py -a resnet --depth 20 --resume checkpoints/cifar10/resnet-20/model_best.pth.tar --evaluate

```
Decompose the trained model with retraining:
```
python cifar-nuclear-regularization.py.py -a resnet --depth 20 --resume checkpoints/cifar10/resnet-20/model_best.pth.tar --evaluate --retrain

```

#### Notes
During decomposition, TRP using value threshold(very small value to truncate singular values) based strategy because the trained model is in low-rank format. Other methods including Channel or spatial-wise decomposition baseline methods use energy threshold.
