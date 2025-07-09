#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dora run -d model=htdemucs \
    ++dset.channels=1 \
    batch_size=8 \
    epochs=150

