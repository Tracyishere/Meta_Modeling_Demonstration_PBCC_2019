#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ~/mysoftware/mmenv3

python incretin_effect.py -ns 50 -p inc -c 0,10,100 -z 0,0,0