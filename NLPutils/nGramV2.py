import torch
import torch.nn as nn
import numpy as np

class nGram(nn.Module):
    def __init__(self, nG=2, inEmbs=300, outEmbs=300, pad=False):
        super().__init__()
        self.gram = nn.Conv1d(inEmbs, outEmbs, kernel_size=(nG), padding=(nG-1) if pad else (0))
        self.padding = pad
    def forward(self, X):
        m,e,n=X.shape
        out = self.gram(X)
        if self.padding:
            out=out[:,:,:n]
        return out


class PhraseLevel(nn.Module):
    def __init__(self, inEmbs=300, outEmbs=300, nGs=[1,2,3], padding=True):
        super().__init__()
        self.nGrams = []
        self.nGs = nGs
        for n in self.nGs:
            print(n)
            self.nGrams.append(nGram(n, inEmbs, outEmbs, pad=padding))
        print(len(self.nGrams))
        
    def forward(self, X):
        out=[]
        for ngram in self.nGrams:
            out += [ngram(X)]
        out = torch.stack(out, dim=2)
        out = torch.max(out, dim=2)[0]
        return out
            
class PhraseLevelMod(nn.Module):
    def __init__(self, inEmbs=300, outEmbs=300, nGs=[1,2,3], padding=True):
        super().__init__()
        self.nGrams = []
        self.nGs = nGs
        for n in self.nGs:
            print(n)
            self.nGrams.append(nGram(n, inEmbs, outEmbs, pad=padding))
        print(len(self.nGrams))
        self.conver = nn.Conv1d(len(nGs)*outEmbs, outEmbs, (1))
        
    def forward(self, X):
        out=[]
        for ngram in self.nGrams:
            out += [ngram(X)]
        out = torch.cat(out, dim=1)
        out = self.conver(out)
        return out

