import os
import re
from analysis.relax import AmberRelaxation
from Bio.PDB import *
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from analysis.relax_utils import PCLIO
from analysis.utils import write_from_string
from analysis.adcp_protocol import adcp_protocol, decompose


RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3
P=PDBParser(QUIET=1)
PI=PCLIO()

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
        fixer.topology,
        fixer.positions,
        open(out_file,"w"),keepIds=True)

def check_GLY_CB(entity):
    for res in entity.get_residues():
        if res.resname == 'GLY' and res.has_id('CB'):
            res.detach_child('CB')
    return entity
    

class FlexPepDock:
    def __init__(self,
                 pdb:str,
                 rec_chain:str,
                 lig_chain:str,
                 rcon_path:str,
                 ori_pdb_path:str,
                 fixed_path:str,
                 relaxed_path:str,
                 docking_path:str,
                 nproc=None,
                 xml=None,
                 soft_relax=True
                 ):
        self.pdb_file = pdb
        self.rec_chain = rec_chain
        self.lig_chain = lig_chain
        self.ori_path = ori_pdb_path
        self.rcon_path = rcon_path
        self.fixed_path = fixed_path
        self.relaxed_path = relaxed_path
        self.docking_path = docking_path
        self.pdb_string_rlxd = None
        self.pdb_string_ori_merged = None
        self.data_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        self.rcon_file = os.path.join(self.rcon_path, f'{self.data_id}_rcon.pdb')
        self.fixed_file = os.path.join(self.fixed_path, f'{self.data_id}_fixed.pdb')

        ori_pdb_name = re.match(r"^(.+?)(?:_sample_\d+(?:_ss)?)?$", self.data_id).group(1)
        self.ori_file = os.path.join(self.ori_path, f'{ori_pdb_name}.pdb')
        self.relaxed_file = os.path.join(self.relaxed_path, f'{self.data_id}.pdb')
        self.redock_file = os.path.join(self.docking_path, f'redock_{self.data_id}_0001.pdb')
        self.json_file_interf = os.path.join(self.docking_path, f'{self.data_id}_interf.json')
        self.json_file_redock_rose = os.path.join(self.docking_path, f'{self.data_id}_redock_rose.json')
        self.json_file_redock_adcp = os.path.join(self.docking_path, f'{self.data_id}_redock_adcp.json')
        self.nproc = nproc
        self.xml = xml
        self.soft_relax = soft_relax
    
    def reconstruct(self, denovo=False):
        s_pcl = P.get_structure('s_pcl', self.pdb_file)[0]
        s_ori = P.get_structure('s_ori', self.ori_file)[0]
        lig = [s_pcl[self.lig_chain]]
        if denovo:
            rec_ori = [i for i in s_ori.get_chains()]
        else:
            rec_ori = [i for i in s_ori.get_chains() if i.id != self.lig_chain]
        ents = lig + rec_ori
        new_ents = [check_GLY_CB(ent) for ent in ents]
        PI.set_structure_multiple(new_ents)
        PI.save(self.rcon_file)
        fix_pdb(self.rcon_file, self.fixed_file)
        
    def relax(self,):
        ABRlx = AmberRelaxation(max_iterations=RELAX_MAX_ITERATIONS,
                                tolerance=RELAX_ENERGY_TOLERANCE,
                                max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
                                use_gpu=False)
        self.pdb_string_rlxd, _,  = ABRlx.process(open(self.fixed_file).read())
        write_from_string(self.pdb_string_rlxd, self.relaxed_file)
    
    def Rosetta_redock(self, flexppd_path):
        rosetta_script_path = os.path.join(os.path.dirname(flexppd_path), os.path.basename(flexppd_path).replace('FlexPepDocking','rosetta_scripts'))
        if self.soft_relax:
            file_to_relax = self.relaxed_file
        else:
            file_to_relax = self.fixed_file
            
        os.system(f'{flexppd_path} -ex1 -ex2aro -use_input_sc -pep_refine -lowres_preoptimize -nstruct 1\
                      -flexpep_score_only -s {file_to_relax} -out:path:all {self.docking_path} -out:file:scorefile_format json -out:prefix redock_\
                          -overwrite -ignore_zero_occupancy false')
        
        os.system(f'{rosetta_script_path} -s {file_to_relax} -in:file:native {self.fixed_file} -parser:protocol {self.xml} -out:path:all {self.docking_path}\
                   -out:file:scorefile_format json -out:prefix interf_ -overwrite -overwrite -ignore_zero_occupancy false')
    
    def ADCP_redock(self,):
        '''
        adcp and AutoDockTools should be pre-installed!
        '''
        if self.soft_relax:
            file_to_relax = self.relaxed_file
        else:
            file_to_relax = self.fixed_file
        l,r = decompose(file_to_relax, 'A', self.docking_path)
        adcp_protocol(l, r, self.nproc)
    
    def interface_analyze(self, rosetta_script_path):
        if self.soft_relax:
            file_to_relax = self.relaxed_file
        else:
            file_to_relax = self.fixed_file
        os.system(f'{rosetta_script_path} -s {file_to_relax} -in:file:native {self.fixed_file} -parser:protocol {self.xml} -out:path:all {self.docking_path}\
                   -out:file:scorefile_format json -out:prefix interf_ -overwrite -overwrite -ignore_zero_occupancy false')

    def __call__(self, app=None, app_path=None, denovo=False):
        
        self.reconstruct(denovo)
        
        if self.soft_relax:
            try:
                self.relax()
            except:
                self.soft_relax = False
            
        if app =='interface_analyzer' and app_path is not None:
            self.interface_analyze(app_path)
        
        elif app == 'flexpepdock' and app_path is not None:
            self.Rosetta_redock(app_path)

        elif app == 'ADCP':         
            self.ADCP_redock()           
                       
