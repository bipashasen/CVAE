#!/bin/bash
#SBATCH --job-name=cvae_ar
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time 4-01:00:00
#SBATCH --signal=B:HUP@600
#SBATCH --mem-per-gpu=30G

source /home2/bipasha31/miniconda3/bin/activate torch

python main.py