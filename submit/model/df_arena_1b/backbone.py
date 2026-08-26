import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Wav2Vec2Model, Wav2Vec2Config
from .conformer import FinalConformer

class DF_Arena_1B(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = Wav2Vec2Model(Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-1b"))
        self.ssl_model.config.output_hidden_states = True
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1280, 1) #1280 for 1b, 1920 for 2b
        self.sig = nn.Sigmoid()


        self.conformer = FinalConformer(emb_size=1280, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4)

        # Learnable attention weights
        self.attn_scores = nn.Linear(1280, 1, bias=False)
    
    def get_attenF1Dpooling(self, x):
        #print(x.shape, 'x shape in attnF1Dpooling')
        logits = self.attn_scores(x)
        weights = torch.softmax(logits, dim=1)  # (B, T, 1)    
        pooled = torch.sum(weights * x, dim=1, keepdim=True)  # (B, 1, D)
        return pooled
    
    def get_attenF1D(self, layerResult):
        poollayerResult = []
        fullf = []
        for layer in layerResult:
            # layer shape: (B, D, T)
            #layery = layer.permute(0, 2, 1)  # (B, T, D)
            layery = self.get_attenF1Dpooling(layer)  # (B, 1, D)
            poollayerResult.append(layery)
            fullf.append(layer.unsqueeze(1))  # (B, 1, D, T)

        layery = torch.cat(poollayerResult, dim=1)      # (B, L, D)
        fullfeature = torch.cat(fullf, dim=1)          # (B, L, D, T)
        return layery, fullfeature

    def forward(self, x):
        out_ssl = self.ssl_model(x.unsqueeze(0)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        y0, fullfeature = self.get_attenF1D(out_ssl.hidden_states) 
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        fullfeature = self.first_bn(fullfeature)
        fullfeature = self.selu(fullfeature)


        output, _ = self.conformer(fullfeature.squeeze(1))


        return output