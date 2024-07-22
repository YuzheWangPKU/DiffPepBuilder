"""Score network module."""
import torch
from torch import nn

from data import all_atom, residue_constants
from model.layers import Linear
from model.ipa_module import IpaScore
from model.embedding import Embedder


def compute_angles(ca_pos, pts):
    batch_size, num_res, num_heads, num_pts, _ = pts.shape
    calpha_vecs = (ca_pos[:, :, None, :] - ca_pos[:, None, :, :]) + 1e-10
    calpha_vecs = torch.tile(calpha_vecs[:, :, :, None, None, :], (1, 1, 1, num_heads, num_pts, 1))
    ipa_pts = pts[:, :, None, :, :, :] - torch.tile(ca_pos[:, :, None, None, None, :], (1, 1, num_res, num_heads, num_pts, 1))
    phi_angles = all_atom.calculate_neighbor_angles(
        calpha_vecs.reshape(-1, 3),
        ipa_pts.reshape(-1, 3)
    ).reshape(batch_size, num_res, num_res, num_heads, num_pts)

    return  phi_angles


class TorsionAngleDecoder(nn.Module):
    """
    Predict torsion angles from the node embeddings.
    """
    def __init__(self, model_conf, num_torsions, eps=1e-8):
        super(TorsionAngleDecoder, self).__init__()

        self.c = model_conf.node_embed_size
        self.eps = eps
        self.num_torsions = num_torsions

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")
        self.linear_final = Linear(
            self.c, self.num_torsions * 2, init="final")

        self.relu = nn.ReLU()


    def forward(self, s):
        s_initial = s  # [batch_size, N_res, C_s]
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)

        s = s + s_initial
        unnormalized_s = self.linear_final(s)

        # Reshape to [batch_size, N_res, num_torsions, 2]
        unnormalized_s = unnormalized_s.view(s.shape[0], s.shape[1], self.num_torsions, 2)
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(unnormalized_s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        normalized_s = unnormalized_s / norm_denom
        norm_denom = norm_denom[..., 0]  # [batch_size, N_res, num_torsions]

        return norm_denom, unnormalized_s, normalized_s


class ResidueTypeDecoder(nn.Module):
    """
    Predict residue types from the node and edge embeddings.
    """
    def __init__(self, model_conf):
        super(ResidueTypeDecoder, self).__init__()
        self.c = model_conf.node_embed_size + model_conf.edge_embed_size
        self.num_res_types = model_conf.embed.num_aatypes
        self.max_ca_dist = model_conf.decode.res_dist_threshold

        self.MLP = nn.Sequential(
            Linear(self.c, self.c, init="relu"),
            nn.ReLU(),
            Linear(self.c, self.c, init="relu"),
            nn.ReLU(),
            Linear(self.c, self.num_res_types, init="glorot")
        )


    def forward(self, s, z, ca_dist):
        ca_dist_mask = ca_dist < self.max_ca_dist
        z = z * ca_dist_mask[..., None]
        z_mean = z.sum(dim=2) / ca_dist_mask.sum(dim=2, keepdim=True).clamp(min=1)

        s_concat = torch.cat([s, z_mean], dim=-1)  # [batch_size, N_res, C_s + C_z]
        s_logits = self.MLP(s_concat)
        
        return s_logits
    
    
class ScoreNetwork(nn.Module):
    """
    Score network module.
    """
    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = IpaScore(model_conf, diffuser)

        self.torsion_pred = TorsionAngleDecoder(model_conf, num_torsions=3)  # 1 \psi angle + 2 \chi angles
        self.aa_type_pred = ResidueTypeDecoder(model_conf)


    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0


    def forward(self, input_feats):
        """
        Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # Frames as [batch, res, 7] tensors
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [batch_size, N_res]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]  # [batch_size, N_res, N_res]

        # Initial embeddings of positonal and relative indices
        init_node_embed, init_edge_embed = self.embedding_layer(
            aatype=input_feats['aatype'],
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
            esm_embed=input_feats.get('esm_embed', None)
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)
        node_embed = model_out['node_embed']
        edge_embed = model_out['edge_embed']

        # Torsion angle prediction
        torsion_norm, _, torsion_pred = self.torsion_pred(node_embed)  # torsion_norm: [batch_size, N_res, 5]
        psi_pred = torsion_pred[..., 0, :]  # [batch_size, N_res, 2]
        chi_pred = torsion_pred[..., 1:3, :]  # [batch_size, N_res, 2, 2]
        
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]  # [batch_size, N_res, 2]
        psi_pred = self._apply_mask(
            psi_pred, gt_psi, 1 - fixed_mask[..., None])
        
        gt_chi = input_feats['torsion_angles_sin_cos'][..., 3:5, :]  # [batch_size, N_res, 2, 2]
        chi_pred = self._apply_mask(
            chi_pred, gt_chi, 1 - fixed_mask[..., None, None])

        # Backbone atom coordinates
        rigids_pred = model_out['final_rigids']
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        atom37_pred = bb_representations[0].to(rigids_pred.device)  # [batch_size, N_res, 37, 3]
        atom14_pred = bb_representations[-1].to(rigids_pred.device)  # [batch_size, N_res, 14, 3]

        # CA distance matrix
        CA_IDX = residue_constants.atom_order['CA']
        ca_pos_pred = atom37_pred[..., CA_IDX, :]  # [batch_size, N_res, 3]
        ca_dist_pred = torch.linalg.norm(
            ca_pos_pred[:, :, None, :] - ca_pos_pred[:, None, :, :], dim=-1)  # [batch_size, N_res, N_res]
        
        # Residue type prediction
        aa_logits_pred = self.aa_type_pred(node_embed, edge_embed, ca_dist_pred)  # [batch_size, N_res, 20]

        # Model prediction output
        pred_out = {
            'psi': psi_pred,
            'chi': chi_pred,
            'torsion_norm': torsion_norm,
            'aa_logits': aa_logits_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
            'rigids': rigids_pred.to_tensor_7(),
            'atom37': atom37_pred,
            'atom14': atom14_pred,
            'ca_dist': ca_dist_pred
        }

        return pred_out

