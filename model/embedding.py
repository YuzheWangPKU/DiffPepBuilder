"""Embedder module."""
import math
import functools as fn
import torch
from torch import nn
from torch.nn import functional as F

from data import utils as du


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [..., N_res, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)

    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1

    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2

    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')

    assert emb.shape == (timesteps.shape[0], embedding_dim)  # [batch_size, D_t]
    
    return emb


class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        # Amino acid type embedding
        aatype_embed_size = self._embed_conf.aatype_embed_size
        node_embed_dims += aatype_embed_size
        edge_in += aatype_embed_size * 2

        # ESM embedding
        if self._embed_conf.use_esm_embed:
            esm_embed_size = self._embed_conf.esm_embed_size
            node_embed_dims += esm_embed_size
            edge_in += esm_embed_size * 2
            if self._embed_conf.raw_esm_size != self._embed_conf.esm_embed_size:
                self.esm_downsampler = nn.Linear(self._embed_conf.raw_esm_size, self._embed_conf.esm_embed_size)

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )
        self.aatype_embedder = nn.Embedding(
            num_embeddings=self._embed_conf.num_aatypes + 1,  # [MASK] type
            embedding_dim=self._embed_conf.aatype_embed_size,
        )


    def _cross_concat(self, feats_1d, batch_size, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([batch_size, num_res**2, -1])


    def forward(
            self,
            aatype,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
            esm_embed=None
        ):
        """
        Embeds a set of inputs

        Args:
            aatype: [..., N_res] Amino acid type for each residue.
            seq_idx: [..., N_res] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: [..., N_res] mask of fixed (motif) residues.
            self_conditioning_ca: [..., N_res, 3] Ca positions of self-conditioning input.
            esm_embed: [..., N_res, D_esm] ESM embedding for each residue.

        Returns:
            node_embed: [batch_size, N_res, D_node]
            edge_embed: [batch_size, N_res, N_res, D_edge]
        """
        batch_size, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))  # [batch_size, N_res, D_t]
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask.unsqueeze(-1)], dim=-1)  # [batch_size, N_res, D_t + 1]
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, batch_size, num_res)]  # [batch_size, N_res**2, (D_t + 1)*2]

        # Positional index features.
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]  # [batch_size, N_res, N_res]
        rel_seq_offset = rel_seq_offset.reshape([batch_size, num_res**2])
        node_feats.append(self.index_embedder(seq_idx))  # [batch_size, N_res, D_idx]
        pair_feats.append(self.index_embedder(rel_seq_offset))  # [batch_size, N_res**2, D_idx]

        # Amino acid type features.
        mask_idx = self._embed_conf.num_aatypes
        aatype = torch.where(fixed_mask == 0, torch.full_like(aatype, mask_idx), aatype)  # [batch_size, N_res]
        aatype_embed = self.aatype_embedder(aatype)  # [batch_size, N_res, D_type]
        node_feats.append(aatype_embed)
        pair_feats.append(self._cross_concat(aatype_embed, batch_size, num_res))  # [batch_size, N_res**2, D_type*2]

        # ESM features.
        if self._embed_conf.use_esm_embed:
            assert esm_embed is not None, "ESM embedding is not provided."
            if self._embed_conf.raw_esm_size != self._embed_conf.esm_embed_size:
                esm_embed = self.esm_downsampler(esm_embed)
            node_feats.append(esm_embed)  # [batch_size, N_res, D_esm]
            pair_feats.append(self._cross_concat(esm_embed, batch_size, num_res))  # [batch_size, N_res**2, D_esm*2]

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )  # [batch_size, N_res, N_res, N_bins]
            pair_feats.append(sc_dgram.reshape([batch_size, num_res**2, -1]))  # [batch_size, N_res**2, N_bins]

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())  # [batch_size, N_res, D_node]
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())  # [batch_size, N_res**2, D_edge]
        edge_embed = edge_embed.reshape([batch_size, num_res, num_res, -1])

        return node_embed, edge_embed

