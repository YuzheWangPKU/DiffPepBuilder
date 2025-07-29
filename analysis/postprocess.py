import os
import shutil
from analysis.amber_minimize import AmberRelaxation
from Bio.PDB import *
from pdbfixer import PDBFixer

os.environ["ABSL_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

from openmm.app import PDBFile
from analysis.postprocess_utils import PCLIO
from analysis.utils import write_from_string
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.rosetta_scripts import RosettaScriptsParser
from pyrosetta.toolbox import py_jobdistributor as jd


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
        os.makedirs(self.postprocess_dir, exist_ok=True)

        self.pdb_string_relaxed = None
        self.pdb_string_ori_merged = None
        self.data_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        self.recon_file = os.path.join(self.postprocess_dir, f"{self.data_id}_recon.pdb")
        self.fixed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_fixed.pdb")
        self.ori_file = self.find_ori_file()
        self.xml = xml
        self.amber_relax = amber_relax
        self.rosetta_relax = rosetta_relax

    def find_ori_file(self):
        """
        Given self.data_id = e.g. "m_8F4G_2.03_fixed_nat_sample_28",
        this will look for, in order:
        m_8F4G_2.03_fixed_nat_sample_28.pdb
        m_8F4G_2.03_fixed_nat_sample.pdb
        m_8F4G_2.03_fixed_nat.pdb
        m_8F4G_2.03_fixed.pdb   ←  finds this one
        and stops as soon as it exists.
        """
        prefix = self.data_id
        while True:
            candidate = os.path.join(self.ori_dir, f"{prefix}.pdb")
            if os.path.isfile(candidate):
                return candidate
            # if there's nothing left to strip, bail out
            if "_" not in prefix:
                break
            # strip off the last “_…” segment and try again
            prefix = prefix.rsplit("_", 1)[0]

        raise FileNotFoundError(f"No original PDB matching '{self.data_id}' found in {self.ori_dir!r}")
    
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
            use_gpu=False
        )
        self.pdb_string_relaxed, _, = ABRlx.process(open(self.pdb_file).read())
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_amber_relaxed.pdb")
        write_from_string(self.pdb_string_relaxed, relaxed_file)
        self.pdb_file = relaxed_file

    def interface_analyze(self):
        init(
            "-mute all "
            "-ignore_zero_occupancy false "
            f"-in:file:native {self.fixed_file} "
            f"-parser:protocol {self.xml}"
        )

        pose = pose_from_pdb(self.pdb_file)
        mover = RosettaScriptsParser().generate_mover(self.xml)
        mover.apply(pose)
        relaxed_file = os.path.join(self.postprocess_dir, f"{self.data_id}_rosetta_relaxed.pdb")
        pose.dump_pdb(relaxed_file)
        self.pdb_file = relaxed_file

        scorefxn  = get_fa_scorefxn()
        scorefxn(pose)
        jd.output_scorefile(
            pose=pose,
            pdb_name=self.ori_file.replace(".pdb", ""),
            current_name=self.data_id,
            scorefxn=scorefxn,
            nstruct=1,
            scorefilepath=os.path.join(self.postprocess_dir, "rosetta_score.sc"),
            json_format=True
        )

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
