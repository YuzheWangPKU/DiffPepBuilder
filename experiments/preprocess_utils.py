""""""
import pyrootutils

# See: https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True
)

import os, json, string, warnings, argparse, copy
import wget
from tqdm import tqdm
import numpy as np
from Bio import PDB
import dataclasses
import esm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import utils as du
from data import errors
from data import parsers
from data import residue_constants

# ---------- receptor info + chain normalization ----------


def read_receptor_info(json_path, receptor_name):
    with open(json_path, "r") as fh:
        data = json.load(fh)
    lower = {k.lower(): v for k, v in data.items()}
    rec = lower.get(receptor_name.lower())
    if not rec:
        return None, None, None
    return rec.get("motif"), rec.get("hotspots"), rec.get("lig_chain")


def _get_chain_ids(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure("", pdb_path)[0]
    return [ch.id for ch in model.get_chains()]


def _build_chain_map(chain_ids, ligand=None):
    U = list(string.ascii_uppercase)
    if ligand:
        mapping = {ligand: "A"}
        pool = iter([c for c in U if c != "A"])
        for cid in chain_ids:
            if cid == ligand:
                continue
            mapping[cid] = next(pool, None) or _spill(mapping.values())
        return mapping
    else:
        if "A" not in chain_ids:
            return {cid: cid for cid in chain_ids}
        mapping = {}
        pool = iter(U[1:])  # B, C, ...
        for cid in chain_ids:
            mapping[cid] = next(pool, None) or _spill(mapping.values())
        return mapping


def _spill(used):
    # support too-many-chains gracefully
    for pool in (string.ascii_lowercase, string.digits):
        for c in pool:
            if c not in used:
                return c
    raise errors.DataError("Too many chains to reindex.")


def renumber_rec_chain(pdb_path, json_path=None, in_place=False, chain_map=None):
    if not chain_map or all(k == v for k, v in chain_map.items()):
        return {}
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    model = next(structure.get_models())
    chains = list(model.get_chains())
    old_ids = [ch.id for ch in chains]

    # collision-safe: temp ids then final ids
    temp_ids = [
        chr(c)
        for c in range(33, 127)
        if chr(c) not in string.ascii_letters + string.digits
    ]
    for ch, tmp in zip(chains, temp_ids):
        ch.id = tmp
    for ch, old in zip(chains, old_ids):
        ch.id = chain_map.get(old, old)

    out_pdb = pdb_path if in_place else pdb_path.replace(".pdb", "_modified.pdb")
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)

    if json_path is not None:
        with open(json_path, "r") as fh:
            data = json.load(fh)
        pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
        key = next((k for k in data if k.lower() == pdb_name.lower()), None)
        if key is not None:
            rec = data[key]

            def _remap_tok(tok):
                tok = tok.strip()
                return chain_map.get(tok[0], tok[0]) + tok[1:] if tok else tok

            def _remap_seq(spec):
                if spec is None:
                    return None
                if isinstance(spec, list):
                    return [_remap_tok(t) for t in spec]
                toks = [t for t in str(spec).split("-") if t]
                return "-".join(_remap_tok(t) for t in toks)

            for field in ("motif", "hotspots"):
                if field in rec:
                    rec[field] = _remap_seq(rec[field])
            if "lig_chain" in rec and rec["lig_chain"] is not None:
                rec["lig_chain"] = chain_map.get(rec["lig_chain"], rec["lig_chain"])
        out_json = (
            json_path if in_place else json_path.replace(".json", "_modified.json")
        )
        with open(out_json, "w") as fh:
            json.dump(data, fh, indent=4)

    return chain_map


def extract_lig_seq(pdb_path, chain_id):
    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure("", pdb_path)[0]
    if chain_id not in model:
        raise errors.DataError(f"Chain {chain_id!r} not found in {pdb_path}")
    seq = []
    for res in model[chain_id]:
        if "CA" in res and PDB.Polypeptide.is_aa(res, standard=True):
            short = residue_constants.restype_3to1.get(res.resname, "X")
            seq.append(short)
    return "".join(seq)


def _split_annots(val):
    if val is None:
        return None
    if isinstance(val, list):
        return [s.strip() for s in val]
    for sep in ("-", ","):
        if sep in str(val):
            return [s.strip() for s in str(val).split(sep) if s.strip()]
    return [str(val).strip()]


def normalize_receptor_inputs(file_path, receptor_info_path=None, peptide_dict=None):
    """
    Apply priority (lig_chain > hotspots > motif), ensure ligand→'A' (or clear 'A' if no ligand),
    and keep JSON/PDB consistent. Returns (lig_chain_str, motif_list, hotspots_list, lig_chain_seq_or_None).
    """
    pdb_name = os.path.basename(file_path).replace(".pdb", "")
    motif, hotspots, lig_chain = (None, None, None)
    if receptor_info_path:
        motif, hotspots, lig_chain = read_receptor_info(receptor_info_path, pdb_name)
    chain_ids = _get_chain_ids(file_path)
    if lig_chain and lig_chain not in chain_ids:
        raise errors.DataError(f"Ligand chain '{lig_chain}' not found in {pdb_name}.")
    chain_map = _build_chain_map(chain_ids, ligand=lig_chain)
    if any(chain_map[c] != c for c in chain_ids):
        renumber_rec_chain(
            file_path, receptor_info_path, in_place=True, chain_map=chain_map
        )
        # re-read after in-place renumber
        if receptor_info_path:
            motif, hotspots, lig_chain = read_receptor_info(
                receptor_info_path, pdb_name
            )

    lig_seq = None
    if lig_chain:  # now guaranteed to be 'A'
        lig_seq = extract_lig_seq(file_path, "A")
        if motif or hotspots:
            warnings.warn(
                f"Both ligand and motif/hotspots provided for {pdb_name}; using ligand 'A'."
            )
        motif_list, hotspots_list = None, None
        return "A", None, None, lig_seq

    if peptide_dict is None:
        raise errors.DataError(
            f"No ligand or peptide sequences for {pdb_name}. Provide at least one."
        )
    # prefer hotspots over motif
    if hotspots:
        hotspots_list = _split_annots(hotspots)
        if motif:
            warnings.warn(
                f"Both motif and hotspots for {pdb_name}; hotspots take priority."
            )
        motif_list = None
    else:
        hotspots_list = None
        motif_list = _split_annots(motif) if motif else None
        
    return None, motif_list, hotspots_list, None


# ---------- residue selection + center (superset) ----------


def resid_unique(res):
    if isinstance(res, str):
        res_ = res.split()
        return f"{res_[1]}_{res_[2]}"
    return f"{res.get_parent().id}{res.id[1]}"


class ResSelector(PDB.Select):
    def __init__(self, res_ids):
        self.res_ids = set(res_ids)

    def accept_residue(self, residue):
        return resid_unique(residue) in self.res_ids


def get_motif_center_pos(
    infile,
    motif=None,
    hotspots=None,
    lig_chain_str=None,
    hotspot_cutoff=8,
    pocket_cutoff=10,
):
    """
    Unified version (ligand OR hotspots OR motif OR none).
    Ported from process_receptor.get_motif_center_pos.
    """
    p = PDB.PDBParser(QUIET=1)
    struct = p.get_structure("", infile)[0]
    out_motif_file = infile.replace(".pdb", "_processed.pdb")

    # Build raw seq for receptor chains (excluding ligand if set)
    seq_rec, mask_rec, chain_id_rec = [], [], []
    for res in struct.get_residues():
        res_name = residue_constants.substitute_non_standard_restype.get(
            res.resname, res.resname
        )
        try:
            float(res_name)
            raise errors.DataError(
                f"Residue name should not be a number: {res.resname}"
            )
        except ValueError:
            pass
        res_short = residue_constants.restype_3to1.get(res_name, "<unk>")
        if res.parent.id != lig_chain_str:
            seq_rec.append(res_short)
            chain_id_rec.append(res.parent.id)
            mask_rec.append(0.0)
    mask_rec = np.array(mask_rec, dtype=float)

    rec_residues = [r for r in struct.get_residues() if r.parent.id != lig_chain_str]
    ref_coords_ca, rec_keep = [], []

    if lig_chain_str:
        lig_ca = [
            r["CA"].coord for r in struct.get_residues() if r.parent.id == lig_chain_str
        ]
        if not lig_ca:
            raise errors.DataError(
                f"Specified ligand chain {lig_chain_str} not found in {os.path.basename(infile)}"
            )
        for i in lig_ca:
            for k, j in enumerate(rec_residues):
                if np.linalg.norm(j["CA"].coord - i) <= hotspot_cutoff:
                    ref_coords_ca.append(j["CA"].coord)
    elif hotspots:
        io = PDB.PDBIO()
        io.set_structure(struct)
        io.save(out_motif_file, select=ResSelector(hotspots))
        ref_struct = p.get_structure("", out_motif_file)[0]
        ref_coords_ca = [r["CA"].coord for r in ref_struct.get_residues()]
    elif motif:
        io = PDB.PDBIO()
        io.set_structure(struct)
        io.save(out_motif_file, select=ResSelector(motif))
        ref_struct = p.get_structure("", out_motif_file)[0]
        ref_coords_ca = [r["CA"].coord for r in ref_struct.get_residues()]

    if ref_coords_ca:
        for i in ref_coords_ca:
            for k, j in enumerate(rec_residues):
                if np.linalg.norm(j["CA"].coord - i) <= pocket_cutoff:
                    rec_keep.append(resid_unique(j))
                    mask_rec[k] = 1
    else:
        warnings.warn(
            f"No ligand/motif/hotspots for {os.path.basename(infile)}; using whole receptor."
        )
        ref_coords_ca = [
            r["CA"].coord for r in struct.get_residues() if r.parent.id != lig_chain_str
        ]
        rec_keep = [resid_unique(r) for r in rec_residues]
        mask_rec[:] = 1

    io = PDB.PDBIO()
    io.set_structure(struct)
    io.save(out_motif_file, select=ResSelector(rec_keep))
    center_pos = np.sum(np.array(ref_coords_ca), axis=0) / len(ref_coords_ca)
    struct = p.get_structure("", out_motif_file)[0]

    # raw_seq_data per chain
    raw_seq_data = {}
    for aa, chain_id, m in zip(seq_rec, chain_id_rec, mask_rec):
        raw_seq_data.setdefault(chain_id, {"seq": "", "mask": []})
        raw_seq_data[chain_id]["seq"] += aa
        raw_seq_data[chain_id]["mask"].append(float(m))

    return struct, center_pos, raw_seq_data


# ---------- featurization ----------


def featurize_structure(structure, center_pos):
    """Chain feature extraction + concat + modeled_idx."""
    struct_chains = {ch.id.upper(): ch for ch in structure.get_chains()}
    struct_feats, complex_len = [], 0
    for ch in struct_chains.values():
        complex_len += sum(1 for _ in ch.get_residues())
    chain_masks, res_count = {}, 0
    for ch_id_str, ch in struct_chains.items():
        ch_id = du.chain_str_to_int(ch_id_str)
        ch_prot = parsers.process_chain(ch, ch_id)
        ch_dict = du.parse_chain_feats(
            dataclasses.asdict(ch_prot), center_pos=center_pos
        )
        struct_feats.append(ch_dict)
        mask = np.zeros(complex_len)
        mask[res_count : res_count + len(ch_dict["aatype"])] = 1
        chain_masks[ch_id_str] = mask
        res_count += len(ch_dict["aatype"])
    feats = du.concat_np_features(struct_feats, False)
    feats["center_pos"] = center_pos
    aatype = feats["aatype"]
    modeled_idx = np.where(aatype != 20)[0]
    if modeled_idx.size == 0:
        raise errors.LengthError("No modeled residues")
    feats["modeled_idx"] = modeled_idx
    return feats, chain_masks, struct_chains, complex_len


class PretrainedSequenceEmbedder(nn.Module):
    """
    Pretrained protein sequence encoder from ESM (Frozen, default esm2_t33_650M_UR50D).
    Protein sequence representations are pre-calculated and saved.
    """
    huggingface_card = {
        "esm2_t48_15B_UR50D": "facebook/esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D": "facebook/esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
        "esm1b_t33_650M_UR50S": "facebook/esm1b_t33_650M_UR50S"
    }

    def __init__(
            self,
            model_name: str = "esm2_t33_650M_UR50D",
            checkpoint_dir: str = f"{root}/experiments/checkpoints",
            truncation_seq_length: int = 2046,
            max_batch_size: int = 16
    ):
        super(PretrainedSequenceEmbedder, self).__init__()
        self.truncation_seq_length = truncation_seq_length
        self.max_batch_size = max_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = self._build_from_local_checkpoint(model_name, checkpoint_dir)
        self.model.eval()


    def _build_from_local_checkpoint(self, model_name: str, checkpoint_dir: str):
        """
        Build model from local checkpoint.

        Enable CPU offloading with FSDP.
        Adapted from: https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
        """
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        if model_name not in self.huggingface_card:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
        model_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        regression_path = os.path.join(checkpoint_dir, f"{model_name}-contact-regression.pt")

        if not os.path.isfile(model_path):
            print(f"Model file {model_path} not found. Downloading...")
            wget.download(f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt", out=model_path)

        if not os.path.isfile(regression_path):
            print(f"Model file {regression_path} not found. Downloading...")
            wget.download(f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt", out=regression_path)
        
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        except Exception:
            with torch.serialization.safe_globals([argparse.Namespace]):
                model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        
        model.to(self.device)

        return model, alphabet


    def forward(self, raw_data: dict):
        """
        Compute embeddings of protein sequences.
        Modified from: https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
        """
        labels, sequences = [], []
        raw_data = copy.deepcopy(raw_data)

        for pdb_name, pdb_data in raw_data.items():
            for chain_id, seq_data in pdb_data.items():
                labels.append((pdb_name, chain_id))
                sequences.append(seq_data["seq"])

        dataset = [(label, seq) for label, seq in zip(labels, sequences)]
        batch_converter = self.alphabet.get_batch_converter(self.truncation_seq_length)
        data_loader = DataLoader(
            dataset, batch_size=self.max_batch_size, collate_fn=batch_converter, shuffle=False
        )
        print(f"Read sequence data with {len(dataset)} sequences")

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader, desc="Processing protein sequence batches")):
                print(
                    f"Processing {batch_idx + 1} of {1 + len(dataset) // self.max_batch_size} batches ({toks.size(0)} sequences)"
                )
                toks = toks.to(self.device, non_blocking=True)
                repr_layers_num = self.model.num_layers

                results = self.model(toks, repr_layers=[repr_layers_num])
                representations = results["representations"][repr_layers_num].to("cpu")

                for i, (pdb_name, chain_id) in enumerate(labels):
                    mask = torch.tensor(raw_data[pdb_name][chain_id]["mask"]).to("cpu")
                    truncate_len = min(self.truncation_seq_length, mask.shape[0])
                    # Call clone on tensors to ensure tensors are not views into a larger representation
                    # See https://github.com/pytorch/pytorch/issues/1995
                    raw_data[pdb_name][chain_id]["esm_embed"] = representations[i, 1 : truncate_len + 1][mask[: truncate_len].bool()].clone()

            concat_embs = {}
            for pdb_name, chains in raw_data.items():
                concat_embs[pdb_name] = torch.cat([chain["esm_embed"] for chain in chains.values()], dim=0)

        return concat_embs
