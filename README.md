# CPE663-mini-transformer-benchmark
Mini Transformer benchmark project for CPE 663: Multilingual and Low Resource NLP

## Course
CPE 663: Multilingual and Low Resource NLP

## Assignment
Major Assignment III – Building a Mini Transformer and Producing a Small Benchmark

## Overview

This project implements a mini Transformer encoder from scratch using PyTorch for a synthetic sequence classification task.

The task is to predict whether the first non-padding token appears again in the second half of the valid sequence.

The implementation includes:

* token embedding
* positional encoding
* scaled dot-product attention
* multi-head self-attention
* Transformer encoder block
* mean pooling classification head
* benchmarking across multiple model variants

Prebuilt Transformer modules such as torch.nn.Transformer and torch.nn.MultiheadAttention were not used.

## Project Structure


mini_transformer_benchmark/
```├── README.md
├── data.py
├── model.py
├── train.py
├── benchmark.py
├── utils.py
└── report.pdf
```

## Model Variants

The benchmark compares four model settings:

* Model A: positional encoding, 1 head, 1 layer
* Model B: positional encoding, 4 heads, 1 layer
* Model C: no positional encoding, 4 heads, 1 layer
* Model D: positional encoding, 4 heads, 2 layers

## How to Run

Install dependencies:

```pip install torch pandas matplotlib numpy```

Run benchmark:

python benchmark.py

## Author
Youn Thiinzar
