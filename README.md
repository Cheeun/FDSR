# FDSR


This code rebuilds the searched architecture from FGNAS based on EDSR original code.

## 1. Code
Clone this repository into any place you want.
```bash
git clone https://github.com/Cheeun/FDSR.git
cd FDSR
```
## 2. Download Data
save DIV2K and Benchmark dataset from 
[benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB) for evaluation
[DIV2K dataset](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB) for training
in directory dataset/
```bash
FDSR
|-- README.md
|-- environment.yml
`-- dataset
    |-- benchmark
    |   |-- Urban100
    |   |   |-- HR
    |   |   |-- LR_bicubic
    |   |   |-- bin
    |   |-- Set5
    |   |-- Set14
    |   |-- B100
    |   |-- bin
    |-- DIV2K
```

## 2. Conda Environment setting
```
conda env create -f environment.yml --name FDSR
```
## 3. Quickstart (Demo)
Train searched architecture 
the searched architecture here is searched on full EDSR with scale 4.
```bash
cd src       # You are now in */FDSR/src
sh train_x2.sh  # For training FDSR_x2 model
sh train_x4.sh  # For training FDSR_x4 model
sh test.sh   # For testing pretrained models # in progress
```
## 4. Settings
Place the dataset as in #2

## 5. Results
Further compressed architectures to be done.

| Name | Baseline | Training FLOPs | Pruned-ratio | Parameters[K] | Set5 | Set14 | B100 | Urban100 |
|  ---  |  ---  | ---       | ---        | ---  |  ---  |  ---  |  ---  |  ---  |
| **baseline FDSR** | full EDSR x4 | 180G | 100% | 38,473 | 32.14 | 28.57 | 27.56 | 25.99 |
| **50G FDSR** | full EDSR x4 | 50G | 25% | 9,296 | 32.11 | 28.55 | 27.55 | 25.95 |
| **6G FDSR** | full EDSR x4 | 6G | 3.3% | 1,245 | 32.07 | 28.53 | 27.53 | 25.91 |
