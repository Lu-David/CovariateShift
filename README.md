# Robust Bias Aware Classifiers

Pytorch implementation of 2014 Liu Paper https://proceedings.neurips.cc/paper/2014/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation and Setup

```sh
git clone git@github.com:Lu-David/CovariateShift.git
cd CovariateShift
conda env create -f environment.yml
conda activate CovariateShift
pip install -e .
```

## Usage
After create conda env, run the following command to get some figures.
```sh
python ./scripts/gaussian_random.py
```
A png of model results will appear in your current directory.