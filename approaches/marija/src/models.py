import torch
import torch.nn as nn
import  torch
from collections import OrderedDict

from src.resnet_models import resnet34

def _load_weights(model, weights_file):
    newmodel = torch.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))
    if weights_file is not None:
        weights = torch.load(weights_file)['state_dict']
        pretrained_dict = {k.replace("module.", ""): v for k, v in weights.items()}
        newmodel.load_state_dict(pretrained_dict, strict=False)
    return newmodel

class BoneNet3d_2H(nn.Module):
    def  __init__(self, pretrained_weights_backbone=None, pretrained_weights=None, seq_len=32):
        super(BoneNet3d_2H, self).__init__()
        model1 = resnet34(sample_input_D=seq_len, sample_input_H=224, sample_input_W=224, num_seg_classes=1)
        self.backbone = _load_weights(model1, pretrained_weights_backbone)
        self.pool = nn.MaxPool3d((1, 28, 28))
        final_feats = seq_len//8
        self.fc11 = nn.Linear(in_features=512*final_feats, out_features=256)
        self.fc12 = nn.Linear(in_features=256, out_features=1)
        self.fc21 = nn.Linear(in_features=512*final_feats, out_features=256)
        self.fc22 = nn.Linear(in_features=256, out_features=1)
        self.Dropout = nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU(inplace=True)
        if pretrained_weights is not None:
            # Load pre-training weights
            state_dict = torch.load(pretrained_weights)['state_dict']  #pretrained_weights
            msg = self.load_state_dict(state_dict, strict=True)
            print(format(msg))

    def forward(self, x):
        """
        x is the shape of (batch_size, 1, seq_len, 224, 224)
        """
        B, C, D, W, H = x.shape
        x = self.backbone(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.Dropout(x)
        x1 = self.fc11(x)
        x1 = self.relu(x1)
        x1 = self.fc12(x1)
        x2 = self.fc21(x)
        x2 = self.relu(x2)
        x2 = self.fc22(x2)
        clf, reg = x1.squeeze(-1), x2.squeeze(-1)
        reg = torch.sigmoid(reg)
        return clf, reg
    
