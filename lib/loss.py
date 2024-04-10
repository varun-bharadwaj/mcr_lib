import torch
import torch.nn as nn
import numpy as np

class LossFunc(nn.Module):
    def __init__(self, e=0.5):
        super().__init__()
        self.e = e
        self.rs = []
        self.rcs = []
        self.losses = []
        
    def R(self, Z):
        d, m = Z.shape
        I = torch.eye(d)
        res = 1/2 * torch.logdet(I + d/(m * self.e ** 2) * Z.matmul(Z.T))
        self.rs.append(res.item())
        return res
        
    def R_c(self, Z, Pi):
        d, m = Z.shape
        k = Pi.shape[0]
        I = torch.eye(d)
        res = 0.
        for j in range(k):
            trace = torch.trace(Pi[j])
            res += trace / (2 * m) * torch.logdet(I + d/(trace * self.e ** 2) * Z.matmul(Pi[j]).matmul(Z.T))
        self.rcs.append(res.item())
        return res
    
    def forward(self, X, Y):
        Z = X.T
        Y = np.asarray(Y)
        num_samples = Y.shape[0]
        num_categories = max(Y) + 1
        Pi = np.zeros((num_categories, num_samples, num_samples))
        for j in range(len(Y)):
            Pi[Y[j], j, j] = 1
            
        Pi = torch.tensor(Pi, dtype=torch.float32)
        res = self.R(Z) - self.R_c(Z, Pi)
        self.losses.append(res.item())
        return -res