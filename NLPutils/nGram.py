import torch.nn as nn
import torch

class OneGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(1),padding=(0))
    def forward(self,X,form='ncl'):
        if form=='nlc':
            X = X.permute(0,2,1)
        elif form=='ncl':
            pass
        else:
            print("IVALID FORMAT nlc or ncl is accepted")
        N,v,n = X.shape
        out = self.conv(X)[:,:,:n]
        if form=='nlc':
            out = out.permute(0,2,1)
        return out 

class BiGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(2),padding=(1))
    def forward(self,X):
        N,v,n = X.shape
        return self.conv(X)[:,:,:n]

class TriGram(nn.Module):
    def __init__(self,nE=300,nH=128):
        super().__init__()
        self.conv = nn.Conv1d(nE,nH,kernel_size=(3),padding=(1))
    def forward(self,X):
        N,v,n = X.shape
        return self.conv(X)[:,:,:n]