# FDSR2


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
Test searched architecture 
the searched architecture here is searched on small EDSR with scale 4, and in small search space.
```bash
cd src       # You are now in */FDSR/src
sh test_x4.sh
```
Results and Figures in /experiment/searched_small_edsr_x4
