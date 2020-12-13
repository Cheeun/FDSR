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
conda activate FDSR
```
## 3. Quickstart (Demo)
Train and test the searched architecture 

```bash
# baseline: Full EDSR model

# for scale 2 
cd FDSR_f_x2/src       # You are now in */FDSR/FDSR_f_x2/src
python main.py --scale 2 --searched_model fdsr_full_x2_3% # training
python main.py --scale 2 --searched_model fdsr_full_x2_3% --test_only # testing pretrained model

# for scale 4
cd FDSR_f_x4/src       # You are now in */FDSR/FDSR_f_x4/src
python main.py --scale 4 --searched_model fdsr_full_x4_3% # training
python main.py --scale 4 --searched_model fdsr_full_x4_3% --test_only # testing pretrained model

```
## 4. Settings
Place the dataset as in #2

## 5. Results
Further compressed architectures to be done.

| Name | Baseline | Training FLOPs | Pruned-ratio | Parameters[K] | Set5 | Set14 | B100 | Urban100 | Inference time* |
|  ---  |  ---  | ---       | ---        | ---  |  ---  |  ---  |  ---  |  ---  | --- |
| **baseline FDSR** | full EDSR x4 | 180G | 100% | 38,473 | 32.14 | 28.57 | 27.56 | 25.99 | 35.0 |
| **3% FDSR** | full EDSR x4 | 6G | 3.3% | 1,245 | 32.07 | 28.53 | 27.53 | 25.91 | 0.07 |
| **3% FDSR** | full EDSR x2 | 23G | 3.3% | 1,206 | 37.27 | 32.87 | 31.64 | 30.32 | 0.23 |

Inference time(sec)* is calculated for a single full HD image (1920x1080)
