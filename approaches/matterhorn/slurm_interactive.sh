#!/bin/bash

srun --ntasks=16 --time=2:00:00 --mem=16G --gres=gpu:1 -p gpu -N 1 --pty bash -l  
