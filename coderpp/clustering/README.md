# Getting started

## System requirements
### Operating system
Ubuntu 18.04.5 LTS
### Software dependencies
Check `environment.yml`.
### Hardware
We are running it on NVIDIA A100 or NVIDIA GeForce RTX 3090

## Installation guide
1. (recommended) Use conda-pack to relocate the conda environment
Tutorial: https://litingchen16.medium.com/how-to-use-conda-pack-to-relocate-your-condo-environment-622b68e077df
Download environment: https://cloud.tsinghua.edu.cn/f/c4248149f5ae4f388766/
Installation time: within 5 mins
2. Use yaml file to create environment
`conda env create -f environment.yml`

## Instruction for use
1. Put your NER vocabulary under `./ner_data/`. Each line of the file is one term.
2. `cd ./utils`, `bash run.sh`.
3. The result is in `./result/final_cluster_res.txt`.

## Demo
1. Run on the data: we've put a demo under `./ner_data/data.txt`. Run the demo with `cd ./utils` and `bash run.sh`.
2. Expected output: you are expected to get the same result as in `./result/final_cluster_res.txt`.
3. Expected run time for demo: 37.438s
