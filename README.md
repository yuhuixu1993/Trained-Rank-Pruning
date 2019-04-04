# Trained-Rank-Pruning
PyTorch code for ["Trained Rank Pruning for Efficient Deep Neural Networks"](https://arxiv.org/abs/1812.02402)<br>
Our code is built based on  [bearpaw](https://github.com/bearpaw/pytorch-classification)<br>
<img src=framework.png width=75%><br>
What's in this repo so far:
 * TRP code for CIFAR-10 experiments
 * Nuclear regularization code for CIFAR-10 experiments
 
#### Simple Examples
```Shell
optional arguments:
  -a                    model
  --depth               layers
  --epoths              training epochs
  -c                    path to save checkpoints
  --gpu-id              specifiy using GPU or not
  --nuclear-weight      nuclear regularization parameter
```
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
## Results

- Results on CIFAR-10:

|Network| Method |    | # Params | FLOPs |Acc|
|:-----|:-------:|:-----:|:--------:|:-----:|:-----:|:-----:|
|Resnet20| Origin |\| 0.27M | 1x|91.74|
|Resnet20| TRP+Nu |Channel| 0.1M | 2.17x|90.50|
|Resnet20| TRP+Nu |spatial| 0.08M | 2.84x |90.62|
### Citation
If you think this work is helpful for your own research, please consider add following bibtex config in your latex file

```Latex
@article{xu2018trained,
  title={Trained Rank Pruning for Efficient Deep Neural Networks},
  author={Xu, Yuhui and Li, Yuxi and Zhang, Shuai and Wen, Wei and Wang, Botao and Qi, Yingyong and Chen, Yiran and Lin, Weiyao and Xiong, Hongkai},
  journal={arXiv preprint arXiv:1812.02402},
  year={2018}
}
