#!/bin/bash

# Create the data directory if it doesn't exist
mkdir -p data
mkdir -p results/data_preprocessing
mkdir -p results/models

# Download and save the train set (v1.1)
wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

# Download and save the dev set (v1.1)
wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
