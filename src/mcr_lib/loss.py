import torch
import numpy as np

def R(Z: torch.Tensor, eps_sq: float) -> torch.Tensor:
    d, m = Z.shape
    I = torch.eye(d)
    log_det_matrix = I + d/(m * eps_sq) * Z.matmul(Z.T)
    res = 1/2 * sum(torch.log(torch.linalg.eigvalsh(log_det_matrix)))
    return res
    
def R_c(Z: torch.Tensor, Pi: torch.Tensor, eps_sq: float) -> torch.Tensor:
    d, m = Z.shape
    k = Pi.shape[0]
    I = torch.eye(d)
    res = 0

    trace = torch.vmap(torch.trace)(Pi)
    log_det_matrix = I + d/(trace * eps_sq) * Z.matmul(Pi).matmul(Z.T)
    res = sum(trace / (2 * m) * sum(torch.log(torch.linalg.eigvalsh(log_det_matrix))))

    return res

def forward(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    eps_sq = 0.25
    Z = X.T
    Y = np.asarray(Y)
    num_samples = Y.shape[0]
    num_categories = max(Y) + 1
    Pi = np.zeros((num_categories, num_samples, num_samples))
    for j in range(len(Y)):
        Pi[Y[j], j] = 1
        
    Pi = torch.from_numpy(Pi).float()
    res = R(Z, eps_sq) - R_c(Z, Pi, eps_sq)
    return -res