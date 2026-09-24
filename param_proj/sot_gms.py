import geoopt
import torch

from param_proj.sw import one_dimensional_Wasserstein


def SMixW(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L=10, p=2):
    d = mu1s.shape[1]
    theta = torch.randn(L, d, device=mu1s.device)
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))
    prod_mu1s = torch.matmul(mu1s, theta.transpose(0, 1))
    prod_mu2s = torch.matmul(mu2s, theta.transpose(0, 1))
    prod_Sigma1s = torch.sqrt(torch.sum(torch.matmul(Sigma1s, theta.transpose(0, 1)) * theta.transpose(0, 1), dim=1))
    prod_Sigma2s = torch.sqrt(torch.sum(torch.matmul(Sigma2s, theta.transpose(0, 1)) * theta.transpose(0, 1), dim=1))
    X = torch.stack([prod_mu1s, torch.log(prod_Sigma1s)], dim=-1)
    Y = torch.stack([prod_mu2s, torch.log(prod_Sigma2s)], dim=-1)
    psi = torch.randn(L, 2, device=mu1s.device)
    psi = psi / torch.sqrt(torch.sum(psi ** 2, dim=1, keepdim=True))

    X_prod = torch.sum(X * psi, dim=-1)
    Y_prod = torch.sum(Y * psi, dim=-1)
    return torch.mean(one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)) ** (1. / p)


def MixSW(mu1s, Sigma1s, mu2s, Sigma2s, a, b, L=10, p=2):
    d = mu1s.shape[1]
    theta = torch.randn(L, d, device=mu1s.device)
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))

    D = theta[:, None] * torch.eye(
        theta.shape[-1],
        device=mu1s.device
    )

    # Random orthogonal matrices
    Z = torch.randn(size=(L, d, d), device=mu1s.device)
    Q, R = torch.linalg.qr(Z)
    lambd = torch.diagonal(R, dim1=-2, dim2=-1)
    lambd = lambd / torch.abs(lambd)
    P = lambd[:, None] * Q

    A = torch.matmul(
        P,
        torch.matmul(D, torch.transpose(P, -2, -1))
    )
    theta = torch.randn(L, d, device=mu1s.device)
    theta = theta / torch.sqrt(torch.sum(theta ** 2, dim=1, keepdim=True))

    prod_mu1s = torch.matmul(mu1s, theta.transpose(0, 1))
    prod_mu2s = torch.matmul(mu2s, theta.transpose(0, 1))
    log_Sigma1s = geoopt.linalg.sym_logm(Sigma1s)
    log_Sigma2s = geoopt.linalg.sym_logm(Sigma2s)
    prod_Sigma1s = (A[None] * log_Sigma1s[:, None]).reshape(Sigma1s.shape[0], L, -1).sum(-1)
    prod_Sigma2s = (A[None] * log_Sigma2s[:, None]).reshape(Sigma2s.shape[0], L, -1).sum(-1)
    X = torch.stack([prod_mu1s, prod_Sigma1s], dim=-1)
    Y = torch.stack([prod_mu2s, prod_Sigma2s], dim=-1)
    psi = torch.randn(L, 2, device=mu1s.device)
    psi = psi / torch.sqrt(torch.sum(psi ** 2, dim=1, keepdim=True))
    X_prod = torch.sum(X * psi, dim=-1)
    Y_prod = torch.sum(Y * psi, dim=-1)
    return torch.mean(one_dimensional_Wasserstein(X_prod, Y_prod, a, b, p)) ** (1. / p)
