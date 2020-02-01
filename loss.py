import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import multivariate_normal

class NLLoss():
    def __init__(self, params, strokes, stroke_masks):
        self.norm_params(params)
        self.strokes = strokes
        self.stroke_masks = stroke_masks

    def norm_params(self, params):
        e_hat, pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = params
        self.e = torch.sigmoid(e_hat)
        self.pi = torch.softmax(pi_hat, dim=-1)
        self.mu1 = mu1_hat
        self.mu2 = mu2_hat
        self.sigma1 = sigma1_hat.exp()
        self.sigma2 = sigma2_hat.exp()
        self.rho = torch.tanh(rho_hat)


    def bivariate_normal_pdf(self, x1, x2):
        z1, z2 = ((x1 - self.mu1) / self.sigma1) ** 2, ((x2 - self.mu2) / self.sigma2) ** 2
        z3 = (2 * self.rho * (x1 - self.mu1) * (x2 - self.mu2)) / (self.sigma1 * self.sigma2)
        z = z1 + z2 - z3
        right_term = (-z / 2 * (1 - self.rho ** 2)).exp()
        left_term = 2 * np.pi * self.sigma1 * self.sigma2 * torch.sqrt(1 - self.rho ** 2)
        return right_term / left_term

    def get_loss(self):
        eps = 1e-5
        x_3, x_1, x_2 = torch.chunk(self.strokes, 3, dim=-1)
        x_3 = x_3.squeeze(-1)
        k_stroke_masks = self.stroke_masks.unsqueeze(-1).expand(-1, -1, 20)
        batch_size = x_3.shape[0]
        bivar_prob = self.bivariate_normal_pdf(x_1.expand(-1, -1, 20), x_2.expand(-1, -1, 20))
        bivar_prob *= k_stroke_masks
        bivar_logprob = torch.sum(
            -((self.pi * bivar_prob) + eps).log(),
            dim=-1
        )
        bern_logprob = F.binary_cross_entropy(self.e, x_3, reduction='none')
        bern_logprob *= self.stroke_masks

        return torch.sum(bivar_logprob + bern_logprob , dim=-1).mean()
