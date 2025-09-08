import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ..layers import Linear

class VectorQuantizer(nn.Module):
    """Basic VQ layer with straight-through estimator."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Define explicitly the discrete embeddings and initialize their weights uniformly. 
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z: torch.Tensor):
        # z: (B, D) or (L, D)
        flat_z = z.view(-1, self.embedding_dim)

        # distances: (L, num_embeddings)
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + torch.sum(self.embedding.weight ** 2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)  # (L,)
        quantized = self.embedding(encoding_indices)

        # straight-through trick
        quantized = flat_z + (quantized - flat_z).detach()
        loss = F.mse_loss(quantized.detach(), flat_z) + self.commitment_cost * F.mse_loss(quantized, flat_z.detach())

        return quantized.view(*z.shape), encoding_indices.view(z.shape[0], -1), loss


class Quantizer2D(nn.Module):
    """Quantizer for event camera (x, y) coordinates."""
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            img_size: Tuple[int, int], 
            hidden_dim: int = 64,
            commitment_cost: float = 0.25
    ):
        super().__init__()
        self.img_size = img_size
        self.encoder = nn.Sequential(
            Linear(2, hidden_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU),
            Linear(hidden_dim, embedding_dim, norm_layer=False, act_layer=False),
        )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, xy: torch.Tensor):
        # xy: (B, 2) with pixel coordinates
        norm_xy = xy.clone().float()
        norm_xy[:, 0] = norm_xy[:, 0] / (self.img_size[0] - 1) * 2 - 1  # normalize x to [-1, 1]
        norm_xy[:, 1] = norm_xy[:, 1] / (self.img_size[1] - 1) * 2 - 1  # normalize y to [-1, 1]

        z = self.encoder(norm_xy)  # (B, embedding_dim)
        q, idx, loss = self.vq(z)
        return q, idx, loss


class Quantizer1D(nn.Module):
    """Quantizer for event camera timestamps."""
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            hidden_dim: int = 64,
            commitment_cost: float = 0.25
            ):
        super().__init__()
        assert num_embeddings > 1, "num_embeddings must be greater than 1"
        self.num_embeddings = num_embeddings
        self.encoder = nn.Sequential(
                Linear(1, hidden_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU),
                Linear(hidden_dim, embedding_dim, norm_layer=False, act_layer=False),
            )
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, t: torch.Tensor):
        norm_t = (t.float() / (self.num_embeddings - 1)) * 2 - 1
        z = self.encoder(norm_t)  # (B, embedding_dim)
        q, idx, loss = self.vq(z)
        return q, idx, loss


if __name__ == "__main__":
    # Example usage
    xyq = Quantizer2D(num_embeddings=256, embedding_dim=32, img_size=(240, 180))
    tq = Quantizer1D(num_embeddings=128, embedding_dim=16, max_time=1e6)
    pq = Quantizer1D(num_embeddings=64, embedding_dim=16, max_time=255) 

    xy = torch.randint(0, 240, (10, 2))  # fake coords
    t = torch.randint(0, int(1e6), (10, 1))  # fake timestamps
    p = torch.randint(0, 180, (10, 1))  # fake polarity values

    q_xy, idx_xy, loss_xy = xyq(xy)
    q_t, idx_t, loss_t = tq(t)
    q_p, idx_p, loss_p = pq(p)

    print("XY quantized:", q_xy.shape, "idx:", idx_xy.shape, "loss:", loss_xy.item())
    print("T quantized:", q_t.shape, "idx:", idx_t.shape, "loss:", loss_t.item())
    print("P quantized:", q_p.shape, "idx:", idx_p.shape, "loss:", loss_p.item())