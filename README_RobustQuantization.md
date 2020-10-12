# Robust Quantization- One Model to Rule Them All

the project includes a zip of 2 sub-projects:
- **distiller_with_kure** - A private branch of **Distiller** - an open-source Python package for neural network compression research. a link to the github: https://github.com/NervanaSystems/distiller
    with Distiller we train the models with KURE to be quantization-robust (either we train a FP32 model or a Quantization Aware Training- QAT model)
    License- This project is licensed under the Apache License 2.0 - see the [LICENSE.md] file in distiller directory for details
- **nn_quantization_pytorch** - LAPQ quantization project- LAPQ is a method for deep neural networks quantization.

## Installation

1. git clone https://github.com/moranshkolnik/RobustQuantization.git or unzip the project: Robust_Quantization_With_KURE
2. Create a Python virtual environment
    a. create env
    ```
    $ virtualenv -p /usr/bin/python3 env
    ```
    b. enable the virtual env
    ```
    $ source env/bin/activate
    ```
3. install Distiller- [for more details go to distiller/README.md file]
    ```
    $ cd distiller_with_kure/distiller
    $ pip3 install -e .
    ```
4. install requirements for nn_quantization - lapq
    ```
    $ cd nn_quantization_pytorch/nn-quantization-pytorch
    $ pip3 install -r requirements.txt
    ```

### Required PyTorch Version

Distiller is tested using the default installation of PyTorch 1.3.1, which uses CUDA 10.1. We use TorchVision version 0.4.2. These are included in Distiller's `requirements.txt` and will be automatically installed when installing the Distiller package as listed above.

If you do not use CUDA 10.1 in your environment, please refer to [PyTorch website](https://pytorch.org/get-started/locally/) to install the compatible build of PyTorch 1.3.1 and torchvision 0.4.2.

## Post-Training-Quantization (PTQ) Experiments:
We apply KURE on a pre-trained model and fine-tune it.
Each model is trained until the kurtosis level converges to its target, and until no significant loss in accuracy is observed.
By doing so, we get a quantization-robust model in full precision which then may be used with different quantization algorithms and bit-widths.
After finishing the training process, we quantize our robust model using LAPQ.
We quantize all layers except the first and last layers.

The following instructions describe how to apply KURE and quantize ResNet-18 model with ImageNet dataset. The same can be applied on any other NN architecture (for example: ResNet-50, MobileNet-V2)
1. apply KURE on a model
```
$ cd distiller_with_kure/distiller/examples/classifier_compression
$ python3 compress_classifier.py -a resnet18 -p 50 -b 256 <ImageNet-Path> --epochs 90 --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8 --pretrained --compress=../baseline_networks/imagenet/lr_scheduler_resnet18.yaml -j 22 --lr 0.001 --vs 0 --gpu 0,1 --kurtosis-mode=avg
```
The Checkpoint of the fine-tuned model is in : distiller_with_kure/distiller/examples/classifier_compression/logs/<checkpoint_dir>

2. Post-Training Quantization with LAPQ method.
This example performs 4-bit quantization of ResNet18 for ImageNet.
```
$ cd nn_quantization_pytorch/nn-quantization-pytorch
$ python3 lapq/layer_scale_optimization_opt.py -a resnet18 --dataset imagenet -b 256 --resume=distiller_with_kure/distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --custom_resnet -bw 4 -ba 4  --min_method Powell -maxi 2 -exp temp -cs 512 --gpu_ids 0  --datapath <ImageNet-Path>
```

### More examples:
the same process can be applied on ResNet-50 (-a resnet50) and MobileNet-V2 (-a mobilenet_v2) and with different quantization settings (bit width, bias correction for the weights)
```
# train with KURE
$ cd distiller_with_kure/distiller/examples/classifier_compression
$ python3 compress_classifier.py -a mobilenet_v2 -p 50 -b 256 <ImageNet-Path> --epochs 50 --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8 --pretrained --compress=../baseline_networks/imagenet/lr_scheduler_resnet18.yaml -j 22 --lr 0.001 --vs 0 --gpu 0,1 --kurtosis-mode=avg
$ python3 compress_classifier.py -a resnet50 -p 50 -b 128 <ImageNet-Path> --epochs 90 --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8 --pretrained --compress=../baseline_networks/imagenet/resnet50_imagenet_base_fp32.yaml -j 22 --lr 0.001 --vs 0 --gpu 0,1 --kurtosis-mode=avg

# quantization
$ cd nn_quantization_pytorch/nn-quantization-pytorch
$ python3 lapq/layer_scale_optimization_opt.py -a resnet18 --dataset imagenet -b 256 --resume=distiller_with_kure/distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --custom_resnet -bw 4 -ba 4  --bcorr_w --min_method Powell -maxi 2 -exp temp -cs 512 --gpu_ids 0  --datapath <ImageNet-Path>
$ python3 lapq/layer_scale_optimization_opt.py -a resnet50 --dataset imagenet -b 256 --resume=distiller_with_kure/distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --custom_resnet -bw 3  --min_method Powell -maxi 2 -exp temp -cs 512 --gpu_ids 0  --datapath <ImageNet-Path>
$ python3 lapq/layer_scale_optimization_opt.py -a mobilenet_v2 --dataset imagenet -b 256 --resume=distiller_with_kure/distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --custom_resnet -bw 6 -ba 6  --bcorr_w --min_method Powell -maxi 2 -exp temp -cs 512 --gpu_ids 0  --datapath <ImageNet-Path>

```

## Quantization-Aware-Training (QAT) Experiments:
KURE may be used with any QAT method to improve the model robustness to different quantizers that may be implemented in different accelerators, for example.
We show results on two QAT methods: DoReFa (https://arxiv.org/abs/1606.06160) and LSQ (https://arxiv.org/abs/1902.08153).
To demonstrate the improvement in robustness with KURE, we show robustness to bit-width and robustness to perturbations in quantization step size parameter in QAT methods.

### Train DoReFa-ResNet-18 with KURE

The following instructions describe how to combine KURE with DoReFa-QAT method to quantize ResNet-18 model to 4-bits (weights and activations) with ImageNet dataset. The same can be applied on any other NN architecture
1. combine KURE with DoReFa-ResNet-18
```
$ cd distiller/examples/classifier_compression
$ python3 compress_classifier.py -a resnet18 -p 50 -b 256 <ImageNet-Path> --epochs 90 --compress=../quantization/quant_aware_train/quant_aware_train_dorefa_a4w4_resnet18_last_relu_null_lr_scheduler.yaml --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8 --kurtosis-mode=avg --pretrained -j 22 --lr 0.001 --vs 0 --gpu 0
```
The Checkpoint of the fine-tuned model is in: distiller/examples/classifier_compression/logs/<checkpoint_dir>

2. Bit-width comparison

a)
```
$ cd distiller/distiller/quantization
$ python3 sensitivity_analysis_different_bit_width_qat_dorefa_models.py -a resnet18 -p 50 -b 256 <ImageNet-Path> --epochs 110 --resume=distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --gpu 0 --effective-test-size 1.0 -exp resnet18_bit_width_dorefa_qat
```


3. Robustness to quantization step size perturbation

a)
```
$ cd distiller/distiller/quantization
$ python3 sensitivity_analysis_qat_dorefa_models.py -a resnet18 -p 50 -b 256 <ImageNet-Path> --epochs 110 --resume=distiller/examples/classifier_compression/logs/<checkpoint_dir>/best.pth.tar --gpu 0 --min_ratio 0.7 --max_ratio 1.3 --gr 21 --effective-test-size 0.1 -exp resnet18_dorefa_qat_w4a4_scale_changes
```

b) results are saved in 'resnet18_dorefa_qat_w4a4_scale_changes' directory. Example for plots can be found in nn_quantization_pytorch/nn-quantization-pytorch/jupyter/NIPS2020_experiments.ipynb

### Train LSQ-ResNet-18 with KURE

The following instructions describe how to combine KURE with LSQ-QAT method to quantize ResNet-50 model to 4-bits (weights only) with ImageNet dataset. The same can be applied on any other NN architecture
1. combine KURE with LSQ-ResNet-50

a)
```
$ cd /nn_quantization_pytorch/nn-quantization-pytorch
```

b)
set __IMAGENET_DEFAULT_PATH in file nn_quantization_pytorch/nn-quantization-pytorch/utils/data.py to your local dataset directory

c)
```
$ python3 quantization/qat/cnn_classifier_train_kurtosis.py -a resnet18 --custom_resnet --dataset imagenet -b 128 --gpu_ids 0 --lr 0.001 --lr_step 20 -q -bw 4 -ba 4 -ep 60 -exp temp --qtype lsq  --w-kurtosis --weight-name all --w-lambda-kurtosis 1.0 --w-kurtosis-target 1.8 --pretrained
```

The Checkpoint of the fine-tuned model is in : nn_quantization_pytorch/nn-quantization-pytorch/mxt-sim/ckpt/resnet18/<checkpoint_dir>

2. model quantization (-evaluate flag) to different bit-widths as well
```
$ python3 quantization/qat/cnn_classifier_train.py -a resnet18 --custom_resnet --dataset imagenet -b 128 --gpu_ids 0 --lr 0.01 --lr_step 20 -q -bw 3 -ep 90 -exp resnet18_lsq_w3 --qtype lsq --evaluate --datapath <ImageNet-Path> --resume=nn_quantization_pytorch/nn-quantization-pytorch/mxt-sim/ckpt/resnet18/<checkpoint_dir>/model_best.pth.tar
```

## Jupyter notebook

in nn_quantization_pytorch/nn-quantization-pytorch/jupyter/NIPS2020_experiments.ipynb you can find plots of the experiments results we showed in the paper.
