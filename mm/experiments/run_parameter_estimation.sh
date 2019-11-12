#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate ~/mysoftware/mmenv3

python parameter_estimation.py -nb 50000 -ns 50 -p mmest -v Gb,meal,static Sb,meal,static alpha,meal,static beta,meal,static K,meal,static G,meal,dynamic S,meal,dynamic k,spt,static Npatch,spt,static Nisg,spt,static Ninsulin,spt,static S,spt,dynamic

