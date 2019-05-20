# MobileNetV3_GPUInference

a simple example for testing MobileNetV3 from [1].

## Usage:
step 1: download [1] and extract it.

step 2: clone or download images and 'predict.py' script in this repo, and run the following code:
```
python predict.py --which_model 'large' --gpu_ids '1'
```
or
```
python predict.py --which_model 'small' --gpu_ids '1'
```

[1] A PyTorch implementation of MobileNetV3 [[link]](https://github.com/xiaolai-sqlai/mobilenetv3)
