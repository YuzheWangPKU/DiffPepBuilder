import os
import re
import shutil
from analysis.amber_minimize import AmberRelaxation
from Bio.PDB import *
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from analysis.postprocess_utils import PCLIO
from analysis.utils import write_from_string
from pyrosetta import init, pose_from_pdb
from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser


RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3

P = PDBParser(QUIET=1)
PI = PCLIO()


def fix_pdb(pdb, out_file):
    fixer = PDBFixer(pdb)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7)
    PDBFile.writeFile(
        fixer.topology, fixer.positions, open(out_file, "w"), keepIds=True
    )


def check_GLY_CB(entity):
    for res in entity.get_residues():
        if res.resname == "GLY" and res.has_id("CB"):
            res.detach_child("CB")
    return entity


class Postprocess:
    def __init__(
        self,
        pdb_file: str,
        lig_chain_id: str,
        ori_dir: str,
        out_dir: str,
        xml=None,
        amber_relax=True,
        rosetta_relax=True
    ):
        self.pdb_file = pdb_file
        self.lig_chain_id = lig_chain_id
        self.ori_dir = ori_dir
        self.out_dir = out_dir
        self.postprocess_dir = os.path.join(out_dir, "postprocess_results")

        self.recon_file = os.path.join(self.postprocess_dir, f"{self.data_id}_recon.pdb")
        self.fixed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_fixed.pdb")

        self.pdb_string_relaxed = None
        self.pdb_string_ori_merged = None
        self.data_id = os.path.splitext(os.path.basename(self.pdb_file))[0]

        ori_pdb_name = re.match(r"^(.+?)(?:_sample_\d+(?:_ss)?)?$", self.data_id).group(1)
        self.ori_file = os.path.join(self.ori_dir, f"{ori_pdb_name}.pdb")
        if not os.path.exists(self.ori_file):
            raise FileNotFoundError(f"Original PDB file not found: {self.ori_file}")
        self.xml = xml
        self.amber_relax = amber_relax
        self.rosetta_relax = rosetta_relax

    def reconstruct(self):
        s_pcl = P.get_structure("s_pcl", self.pdb_file)[0]
        s_ori = P.get_structure("s_ori", self.ori_file)[0]
        lig = [s_pcl[self.lig_chain_id]]
        rec_ori = [i for i in s_ori.get_chains() if i.id != self.lig_chain_id]
        ents = lig + rec_ori
        new_ents = [check_GLY_CB(ent) for ent in ents]
        PI.set_structure_multiple(new_ents)
        PI.save(self.recon_file)
        fix_pdb(self.recon_file, self.fixed_file)
        self.pdb_file = self.fixed_file

    def relax(self):
        ABRlx = AmberRelaxation(
            max_iterations=RELAX_MAX_ITERATIONS,
            tolerance=RELAX_ENERGY_TOLERANCE,
            max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
            use_gpu=False,
        )
        self.pdb_string_relaxed, _, = ABRlx.process(open(self.pdb_file).read())
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_amber_relaxed.pdb")
        write_from_string(self.pdb_string_relaxed, relaxed_file)
        self.pdb_file = relaxed_file
            
    def interface_analyze(self):
        init(
            "-mute all "
            f"-in:file:native {self.fixed_file} "
            f"-parser:protocol {self.xml}"
        )
        pose = pose_from_pdb(self.pdb_file)
        parser = RosettaScriptsParser()
        mover = parser.generate_mover(self.xml)
        mover.apply(pose)
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_rosetta_relaxed.pdb")
        pose.dump_pdb(relaxed_file)
        self.pdb_file = relaxed_file

    def __call__(self):
        try:
            self.reconstruct()
        except Exception as e:
            raise RuntimeError(f"Error during reconstruction: {e}")
        if self.amber_relax:
            try:
                self.relax()
            except Exception as e:
                print(f"Skipping Amber relaxation due to error: {e}")
        if self.rosetta_relax:
            try:
                self.interface_analyze()
            except Exception as e:
                print(f"Skipping Rosetta relaxation due to error: {e}")
        shutil.copy(self.pdb_file, os.path.join(self.out_dir, f"{self.data_id}_final.pdb"))
