![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# MiceBoneChallenge: Micro-CT public dataset and six solutions for automatic growth plate detection in micro-CT mice bone scans

## Description of the project 

The repository has the source code for 6 solutions developed by 6 teams in Anonymous Company internal challenge on detecting the growth plate plane index (GPPI) in 3D micro-CT mice bones.

For the challenge, we prepared and annotated a unique high quality micro-CT 3D bone imaging dataset from 83 mice [[dataset](data/data.md)]. 

We will release all training and test data to facilitate reproducibility and farther model development.

## How to use and run the code

The code from the teams has both training scripts as well as scripts for the inference using the pretrained solutions. 
The approaches per team are in `../approaches/teamname`

The six approaches are from the following six teams:
  - `SafetyNNet` or SN team [[description](approaches/safetynnet/README.md)][[code](approaches/safetynnet/)][[model](models/models.md)];
  - `Matterhorn` or MH team [[description](approaches/matterhorn/README.md)][[code](approaches/matterhorn/)][[model](models/models.md)];
  - `Exploding Kittens` or EK team [[description](approaches/explodingkittens/README.md)][[code](approaches/explodingkittens/)][[model](models/models.md)];
  - `Code Warriors 2` or CW team [[description](approaches/code-warriors2/README.md)][[code](approaches/code-warriors2/)][[model](models/models.md)];
  - `Subvisible` or SV team [[description](approaches/subvisible/README.md)][[code](approaches/subvisible/)][[model](models/models.md)];
  - `Byte me if you can` or BM team [[description](approaches/bytemeifyoucan/README.md)][[code](approaches/bytemeifyoucan/)][[model](models/models.md)];

## Software requirements 

The following requirements are for all six approaches:

Linux platforms are supported - as long as the dependencies are supported on these platforms.


Anaconda or miniconda with Python 3.9 - 3.11

The tool has been developed on a Linux platform.

Python libraries and their versions are in requirements.txt

## Licence 


The software is licensed under the MIT license (see LICENSE file), and is free and provided as-is.


