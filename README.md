# MiceBoneChallenge: Micro-CT public dataset and six solutions for automatic growth plate detection in micro-CT mice bone scans


The repository has the source code for 6 solutions developed by 6 teams in Anonymous Company internal challenge on detecting the growth plate plane index (GPPI) in 3D micro-CT mice bones.

For the challenge, we prepared and annotated a unique high quality micro-CT 3D bone imaging dataset from 83 mice [[dataset](models.md)]. 

We will release all training and test data to facilitate reproducibility and farther model development.

The code from the teams has both training scripts as well as scripts for the inference using the pretrained solutions. 
The approaches per team are in `../approaches/teamname`

The six approaches are from the following six teams:
  - `SafetyNNet` or SN team [[description](approaches/safetynnet/README.md)][[code](approaches/safetynnet/)][[model](models.md)];
  - `Matterhorn` or MH team [[description](approaches/matterhorn/README.md)][[code](approaches/matterhorn/)][[model](models.md)];
  - `Exploding Kittens` or EK team [[description](approaches/explodingkittens/README.md)][[code](approaches/explodingkittens/)][[model](models.md)];
  - `Code Warriors 2`or CW team [[description](approaches/code-warriors2/README.md)][[code](approaches/code-warriors2/)][[model](models.md)];
  - `Subvisible` or SV team [[description](approaches/subvisible/README.md)][[code](approaches/subvisible/)][[model](models.md)];
  - `Byte me if you can` or BM team [[description](approaches/bytemeifyoucan/README.md)][[code](approaches/bytemeifyoucan/)][[model](models.md)];


