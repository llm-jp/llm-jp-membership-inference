#!/bin/bash
#SBATCH --job-name=NoisyGPTNeoX
#SBATCH --partition=010-partition
#SBATCH --gres=gpu:8               # 请求8个GPU
#SBATCH --nodes=1                  # 确保在一个节点上运行
#SBATCH --ntasks=8                 # 并行启动8个任务
#SBATCH --output=test.log
#SBATCH --error=test.err
#SBATCH --cpus-per-task=8          # 每个任务使用的CPU数量


python test.py