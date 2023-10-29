# Composite transformer multi-stage defect detection for sewer pipes

# Demo

<img src="demo\demo.gif" alt="demo"  />

## Environments

- windows 10
- GPU RTX8000×1
- cuda 11.1
- cudnn 8
- python 3.7
- pytorch 1.8.1
- mmcv-full 1.4.0
- yapf 0.32.0
- albumentations 1.3.0
- apex 0.1 (optional)

## Installation

```
pip install -v -e .
```

## Offline data augmentation

Brightness adjustment and Sharpen offline data augmentation.

```
python projects\dataaug.py 
```

## Model train

It is recommended that the dataset be converted to an annotation file in coco format.

```
#Automatic mixed-precision training
python tools\train.py projects\cbnet\cbswinl_cascade.py <input_images_path> <path_of_modified _contrast_images> <path_of_offline_images>
#Without Automatic mixed-precision training
python tools\train.py projects\cbnet\cbswinl_cascade_woamp.py <input_images_path> <path_of_modified _contrast_images> <path_of_offline_images>
```

## Model inference 

```
#With out TTA
python tools\test.py projects\cbnet\cbswinl_cascade.py work_dirs\cbswinl_cascade\latest.pth --eval bbox
#With TTA
python tools\test.py projects\cbnet\cbswinl_cascade_tta.py work_dirs\cbswinl_cascade\latest.pth --eval bbox
```

## Model box fusion

Note the paths to the prediction file on line 19 and the result file on line 89 of wbf.py. The final file is wbf_result.json.

```
python projects\wbf.py
```

