#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1

python train.py
