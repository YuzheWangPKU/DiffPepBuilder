#!/usr/bin/env python

import pyrootutils

# See: https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True
)

import os
from typing import List
import numpy as np
from numpy import dot, transpose, sqrt
from numpy.linalg import svd, det

EPS = 1e-8


AA = {'G': 'GLY', 'A': 'ALA', 'V': 'VAL', 'L': 'LEU','I': 'ILE',
      'F': 'PHE', 'W': 'TRP', 'Y': 'TYR', 'D': 'ASP','N': 'ASN', 
      'E': 'GLU', 'K': 'LYS', 'Q': 'GLN', 'M': 'MET', 'S': 'SER',
      'T': 'THR', 'C': 'CYS', 'P': 'PRO', 'H': 'HIS', 'R': 'ARG'}

aa = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE':'I', 
      'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'ASP': 'D', 'ASN': 'N',
      'GLU': 'E', 'LYS': 'K', 'GLN': 'Q', 'MET': 'M','SER': 'S', 
      'THR': 'T', 'CYS': 'C', 'PRO': 'P', 'HIS': 'H', 'ARG': 'R', 
      'HSD': 'H', 'CYX': 'C'}


# Classes------------------------------------------------------------------------------------------------
'''Classes of Structure algorithms'''

class Structure:
    def __init__(self, file, **kwargs):
        self.file_pn = file
        self.pdbid = os.path.basename(os.path.splitext(file)[0])
        if 'model' in kwargs :
            self.pdb_atoms,self.pdb_hetatoms,self.title = self.get_pdb_atoms(model=kwargs['model'])
        elif 'atoms' in kwargs:
            self.pdb_atoms = kwargs['atoms']
            self.title = ''
        elif 'chains' in kwargs:
            self.pdb_atoms, self.pdb_hetatoms, self.title = self.get_pdb_atoms(chains=kwargs['chains'])    
        else:
            self.pdb_atoms, self.pdb_hetatoms, self.title = self.get_pdb_atoms()
        self.chains = self.get_pdb_chains()
        self.residues = self.get_pdb_residues(sort=True)
        self.coords = self.get_pdb_coords()
        self.seq = self.get_pdb_seq()
        self.sort_atoms_by_residue()
        self.fullfill_occ()
        self.ss_terminal = []
        self.mode = None

    def get_pdb_atoms(self, **kwargs):
        f = open(self.file_pn)
        pdblines = f.readlines()
        f.close()
        if 'model' in kwargs:
            lines_new = []
            for line in pdblines:
                lines_new.append(line)
                if line.strip() == 'ENDMDL':
                    break
            pdblines = lines_new
        
        if 'chains' in kwargs:
            lines_new = []
            for line in pdblines:
                if len(line) < 30:
                    continue
                if line[21] in kwargs['chains']:
                    lines_new.append(line)
            pdblines = lines_new
        pdb_atoms = []
        pdb_hetatoms = []
        title = []
        for idx in range(len(pdblines)):
            if idx != 0:
                idb = pdblines[idx - 1][17:20].strip() + '_' + pdblines[idx - 1][12:16].strip()
            else:
                idb = ''
            idn = pdblines[idx][17:20].strip() + '_' + pdblines[idx][12:16].strip()
            if pdblines[idx][0:4] == 'ATOM':
                if idn != idb and pdblines[idx][12:16].strip() != 'OXT' and pdblines[idx][26] == ' ':
                    # print(''.join([i for i in pdblines[idx][12:16].strip() if not i.isdigit()])[0])
                    if ''.join([i for i in pdblines[idx][12:16].strip() if not i.isdigit()])[0] != 'H':
                        pdb_atoms.append(pdblines[idx].strip() + '\n')
            elif pdblines[idx][0:6] == 'HETATOM':
                pdb_hetatoms.append(pdblines[idx].strip()+'\n')
            elif pdblines[idx][0:6] == 'HEADER' or pdblines[idx][0:6] == 'REMARK':
                title.append(pdblines[idx])
   
        return pdb_atoms, pdb_hetatoms, title

    def get_pdb_coords(self):
        coords = {}
        for atom in self.pdb_atoms:
            idt = atom[21] + '_' + atom[22:26].strip() + '_' + aa[atom[17:20].strip()] + '_' + atom[12:16].strip()
            # '''chainID_residueID_residueSEQ_atomID'''
            x = float(atom[30:38].strip())
            y = float(atom[38:46].strip())
            z = float(atom[46:54].strip())
            coords[idt] = np.array([x, y, z])
        return coords

    def get_pdb_residues(self,**kwargs):
        residues = {}
        for atom in self.pdb_atoms:
            # print(atom)
            idt = atom[21] + '_' + atom[22:26].strip() + '_' + aa[atom[17:20].strip()]
            if idt not in residues:
                residues[idt] = []
                residues[idt].append(atom)
            else:
                residues[idt].append(atom)
        sorted_residues = {}
        new_residues = {}
        if kwargs != {}  and (not len(self.chains) >= 2):
            if kwargs['sort'] == True:
                for i in sorted(residues):
                    # print(i)
                    sorted_residues[i] = residues[i]
            new_residues = {}
            order = sorted([int(i.split('_')[1]) for i in sorted_residues])
            # print(order)
            for i in order:
                for j in sorted_residues:
                    if int(j.split('_')[1]) == i:
                        new_residues[j] = sorted_residues[j]
            return new_residues
        return residues

    def get_pdb_chains(self):
        atoms = self.pdb_atoms
        chains = {}
        for atom in atoms:
            if atom[21] not in chains:
                chains[atom[21]] = []
                chains[atom[21]].append(atom)
            else:
                chains[atom[21]].append(atom)
        return chains

    def get_pdb_seq(self):
        seq = ''
        for key in self.residues.keys():
            seq += key[-1]
        return seq

    def sort_atoms_by_residue(self):
        for i in self.residues:
            mca = ['N', 'CA', 'C', 'O']
            residue = self.residues[i].copy()
            # print(residue)
            for j in self.residues[i]:
                element = j[12:16].strip()
                if element in mca:
                    # print(element)
                    idx = mca.index(element)
                    mca[idx] = j
                    residue.remove(j)
            # print(mca)
            preserve = False
            for atom in mca:
                if len(atom) <= 3:
                    preserve = True
                    break
            if preserve == True:
                self.residues[i] = residue
                continue
            self.residues[i] = mca + residue
        newa = []
        for i in self.residues:
            newa += self.residues[i]
        self.pdb_atoms = newa

    def update_pdb_coords(self, newc, show=False):
        for i in range(len(self.pdb_atoms)):
            if show:
                print(newc[i])
            self.pdb_atoms[i] = self.pdb_atoms[i][:30] + str(newc[i][0]).rjust(8) + \
                                str(newc[i][1]).rjust(8) + str(newc[i][2]).rjust(8) + self.pdb_atoms[i][54:]
            # print(self.pdb_atoms[i])
        # print(self.pdb_atoms)
       
        self.residues = self.get_pdb_residues()
        self.coords = self.get_pdb_coords() 
        self.chains = self.get_pdb_chains()      
        self.seq = self.get_pdb_seq()

    def norm_pdb_atoms(self, chain_name):
        for i in range(len(self.pdb_atoms)):
            self.pdb_atoms[i] = self.pdb_atoms[i][:6] + str(i + 1).rjust(5) + self.pdb_atoms[i][11:21] + chain_name + \
                                self.pdb_atoms[i][22:]
        
        self.residues = self.get_pdb_residues()
        self.coords = self.get_pdb_coords()
        self.chains = self.get_pdb_chains()
        self.seq = self.get_pdb_seq()

    def norm_pdb_residues(self):
        keys = [i for i in self.residues.keys()]
        for i in range(len(keys)):
            for j in range(len(self.residues[keys[i]])):
                self.residues[keys[i]][j] = self.residues[keys[i]][j][:22] + str(i + 1).rjust(4) + \
                                            self.residues[keys[i]][j][26:]
        
        self.pdb_atoms = residues2atoms(self.residues)
        self.coords = self.get_pdb_coords()
        self.chains = self.get_pdb_chains()
        self.seq = self.get_pdb_seq()
    
    def fullfill_occ(self,):
        for i in range(len(self.pdb_atoms)):
            self.pdb_atoms[i] = self.pdb_atoms[i][:55] + ' 1.00' + '  1.00' + self.pdb_atoms[i][66:]

    def dump(self,filepath):
        # self.norm_pdb_atoms(chain)
        f = open(filepath, 'w')
        for atom in self.pdb_atoms:
            f.write(atom)
        f.close()

    def __repr__(self):
        if self.mode:
            return self.mode[0] + self.mode[1]
        return self.pdbid
    

class SVDSuperimposer:
    def __init__(self):
        """Initialize the class."""
        self._clear()

    # Private methods

    def _clear(self):
        self.reference_coords = None
        self.coords = None
        self.transformed_coords = None
        self.rot = None
        self.tran = None
        self.rms = None
        self.init_rms = None

    def _rms(self, coords1, coords2):
        diff = coords1 - coords2
        return sqrt(sum(sum(diff * diff)) / coords1.shape[0])

    # Public methods

    def set(self, reference_coords, coords):
        # clear everything from previous runs
        self._clear()
        # store coordinates
        self.reference_coords = reference_coords
        self.coords = coords
        n = reference_coords.shape
        m = coords.shape
        if n != m or not (n[1] == m[1] == 3):
            raise Exception("Coordinate number/dimension mismatch.")
        self.n = n[0]

    def run(self):
        """Superimpose the coordinate sets."""
        if self.coords is None or self.reference_coords is None:
            raise Exception("No coordinates set.")
        coords = self.coords
        reference_coords = self.reference_coords
        # center on centroid
        av1 = sum(coords) / self.n
        av2 = sum(reference_coords) / self.n
        coords = coords - av1
        reference_coords = reference_coords - av2
        # correlation matrix
        a = dot(transpose(coords), reference_coords)
        u, d, vt = svd(a)
        self.rot = transpose(dot(transpose(vt), transpose(u)))
        # check if we have found a reflection
        if det(self.rot) < 0:
            vt[2] = -vt[2]
            self.rot = transpose(dot(transpose(vt), transpose(u)))
        self.tran = av2 - dot(av1, self.rot)

    def get_transformed(self):
        """Get the transformed coordinate set."""
        if self.coords is None or self.reference_coords is None:
            raise Exception("No coordinates set.")
        if self.rot is None:
            raise Exception("Nothing superimposed yet.")
        if self.transformed_coords is None:
            self.transformed_coords = dot(self.coords, self.rot) + self.tran
        return self.transformed_coords

    def get_rotran(self):
        """Right multiplying rotation matrix and translation."""
        if self.rot is None:
            raise Exception("Nothing superimposed yet.")
        return self.rot, self.tran

    def get_init_rms(self):
        """Root mean square deviation of untransformed coordinates."""
        if self.coords is None:
            raise Exception("No coordinates set yet.")
        if self.init_rms is None:
            self.init_rms = self._rms(self.coords, self.reference_coords)
        return self.init_rms

    def get_rms(self):
        """Root mean square deviation of superimposed coordinates."""
        if self.rms is None:
            transformed_coords = self.get_transformed()
            self.rms = self._rms(transformed_coords, self.reference_coords)
        return self.rms


class SS_geo:
    def __init__(self, s, index_preserve:List[int]):
        CAs = {}
        CBs = {}
        self.geo_as = {}
        for i in s.coords:
            m = i.split('_')
            if int(m[1]) in index_preserve:
                continue
            idt = m[0] + '_' + m[1] + '_' + m[2]
            if i.split('_')[-1] == 'CA':
                CAs[idt] = s.coords[i]
            elif i.split('_')[-1] == 'CB':
                CBs[idt] = s.coords[i]
        self.geo = {}
        for idx, i in enumerate(CAs):
            for j in CBs:
                if i == j:
                    self.geo_as[i] = [CAs[i],CBs[j]]
        for i in self.geo_as:
            for j in self.geo_as:
                if int(i.split('_')[1]) + 2 <= int(j.split('_')[1]):
                    self.geo[i+'@'+j] = self._compute_geo(self.geo_as[i], self.geo_as[j])
        # print(self.geo)

    def _vectors(self, p1, p2, p3, p4):
        v1 = p1 - p2
        v2 = p2 - p4
        v3 = p4 - p3
        # print(v1,v2,v3)
        return v1, v2, v3

    def _compute_distance(self, v):
        return np.linalg.norm(v)

    def _compute_angle(self, v1, v2):
        cosin = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(cosin))

    def _compute_dihedral(self, v1, v2, v3):
        axis1 = np.cross(v1, v2)
        axis2 = np.cross(v2, v3)
        cosin = axis1.dot(axis2) / (np.linalg.norm(axis1) * np.linalg.norm(axis2))
        # print(axis1,axis2)
        return np.degrees(np.arccos(cosin))

    def _compute_geo(self,geoa1,geoa2):
        v1, v2, v3 = self._vectors(geoa1[0], geoa1[1], geoa2[0], geoa2[1])
        distance = self._compute_distance(v2)
        angle1 = self._compute_angle(v1, v2)
        angle2 = self._compute_angle(v2, v3)
        dihedral = self._compute_dihedral(v1, v2, v3)
        return distance, angle1, angle2, dihedral


# Superimpose---------------------------------------------------------------------------------------------
'''Superpose Functions of Structure algorithms'''

def superimpose(cs1, cs2):
    imposer = SVDSuperimposer()
    imposer.set(cs1, cs2)
    imposer.run()
    rot, tran = imposer.get_rotran()
    rms = imposer.get_rms()
    return rot, tran, rms

def dump(newa, filepath):
    f = open(filepath, 'w')
    for atom in newa:
        f.write(atom)
    f.close()

def get_coords(pdb_atoms):
    coords = []
    for atom in pdb_atoms:
        x = float(atom[30:38].strip())
        y = float(atom[38:46].strip())
        z = float(atom[46:54].strip())
        coords.append([x, y, z])
    return np.array(coords)

def get_coord(atom):
    x = float(atom[30:38].strip())
    y = float(atom[38:46].strip())
    z = float(atom[46:54].strip())
    coords = [x, y, z]
    return np.array(coords)

def update_coords(pdb_atoms, newc):
    for i in range(len(pdb_atoms)):
        pdb_atoms[i] = pdb_atoms[i][:30]+str(newc[i][0]).rjust(8)+str(newc[i][1]).rjust(8)+str(newc[i][2]).rjust(8)+pdb_atoms[i][54:]


def cal_distance(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

def atomfilter(atomtype, coords):
    target_set = []
    for i in coords.keys():
        if i.split('_')[-1] == atomtype:
            target_set.append(coords[i])
    return np.array(target_set)

def atomfilter_t(atomtype, coords):
    target_set = []
    for i in coords.keys():
        if i.split('_')[-1] == atomtype:
            target_set.append(coords[i])
    target_set.reverse()
    return np.array(target_set)

def dict2matrix(dict_name):
    return np.array([i for i in dict_name.values()])

def residues2atoms(residues):
    atoms = []
    for i in residues:
        for j in residues[i]:
            atoms.append(j)
    return atoms

def norm_atoms(atoms,chain):
    for i in range(len(atoms)):
        atoms[i] = atoms[i][:6] + str(i+1).rjust(5) + atoms[i][11:21] + chain + atoms[i][22:]

def norm_residues(residues):
    keys = [i for i in residues.keys()]
    for i in range(len(keys)):
        for j in range(len(residues[keys[i]])):
            residues[keys[i]][j] = residues[keys[i]][j][:22]+str(i).rjust(4)+residues[keys[i]][j][26:]


def cys_terminal(s, ss_prob, bf_th):
    # print(s.pdb_atoms)
    assert len(s.seq) == len(ss_prob) 
    index_preserve = [i for i in range(len(ss_prob)) if ss_prob[i] <= bf_th]
    # print(index_preserve)
    ss_geo = SS_geo(s, index_preserve).geo
    # print(ss_geo)
    ss_terminal = []
    for i in ss_geo:
        # print(ss_geo[i][0])
        if 3.0 <= ss_geo[i][0] <= 4.5 and 0 <= ss_geo[i][1] <= 120 and 0 <= ss_geo[i][2] <= 120: # and (0 <= ss_geo[i][3] <= 60 or 120 <= ss_geo[i][3] <= 180):
            ss_terminal.append(i)
    # print(ss_terminal)
    # ss_terminal = ['A_3_G@A_9_G', 'A_4_L@A_9_G', 'A_5_L@A_9_G', 'A_7_G@A_9_G']
    s.ss_terminal=[]
    for i in ss_terminal:
        sim_ss =  compare_ss(i, s.ss_terminal)
        if sim_ss != []:
            bfsi = cal_bfs_ss(i, ss_prob)
            for j in sim_ss:
                bfsj = cal_bfs_ss(j, ss_prob)
                # print(bfsi, bfsj)
                if bfsi >= bfsj:
                    s.ss_terminal.remove(j)
                    if i in  s.ss_terminal:
                        continue
                    s.ss_terminal.append(i)
                    # s.ss_terminal.remove(j)
                else:
                    continue
        else:
            s.ss_terminal.append(i)
    if s.ss_terminal:
        print(s.ss_terminal)
        return True
    return False

def compare_ss(st1,ss_all):
    sts = []
    for st in ss_all:
        for i in st1.split('@'):
            for j in st.split('@'):
                if i==j:
                    sts.append(st)
    return sts

def cal_bfs_ss(st, ss_prob):
    sts = st.split('@')
    return ss_prob[int(sts[0].split('_')[1])-1] + ss_prob[int(sts[1].split('_')[1])-1]

def read_ssdata(datapath):
    database = {}
    datapath = [os.path.join(datapath,i) for i in os.listdir(datapath)]
    for i in datapath:
        database[i] = Structure(i)
    return database

# Builder---------------------------------------------------------------------------------------------

def build_ssbond(s:Structure, database, max_num, lig_ch):
    ss_residues = []
    count = 0
    rms_min = 1000000
    # print(list(s.coords.keys()))
    ss_residues_ind = []
    for i in s.ss_terminal:
        ss_residues.append(i.split('@'))
        ss_residues_ind += i.split('@')
    for i in ss_residues:
        
        cs1 = np.array([s.coords[i[0]+'_CA'], s.coords[i[1]+'_CA'], s.coords[i[0]+'_CB'], s.coords[i[1]+'_CB']])
        for ss in database:
            ss = database[ss]
            CAs = []
            CBs = []
            for j in ss.coords:
                if j.split('_')[-1] == 'CA':
                    CAs.append(ss.coords[j])
                elif j.split('_')[-1] == 'CB':
                    CBs.append(ss.coords[j])
            cs2 = np.array(CAs + CBs)
            rot, tran, rms = superimpose(cs1, cs2)
            if rms < rms_min:
                rms_min = rms
                rs_picked = i
                ss_picked = ss
                transformed_coords = (dot(dict2matrix(ss.coords), rot) + tran).round(2)
                ss.update_pdb_coords(transformed_coords)
        # print(rs_picked[0])
        s.residues[rs_picked[0]] = s.residues[rs_picked[0]][0:4]
        s.residues[rs_picked[1]] = s.residues[rs_picked[1]][0:4]
        ss_keys = [i for i in ss_picked.residues.keys()]
        s.residues[rs_picked[0]] += ss_picked.residues[ss_keys[0]][4:]
        s.residues[rs_picked[1]] += ss_picked.residues[ss_keys[1]][4:]

        for i in range(6):
            sc = {'4':'C', '5':'S' }
            if i < 4:
                atom0 = s.residues[rs_picked[0]][i]
                atom1 = s.residues[rs_picked[1]][i]
                atom0 = atom0[0:17] + 'CYS' + atom0[20:]
                s.residues[rs_picked[0]][i] = atom0
                atom1 = atom1[0:17] + 'CYS' + atom1[20:]
                s.residues[rs_picked[1]][i] = atom1
            elif i >= 4:
                atom_norm0 = s.residues[rs_picked[0]][0]
                atom_norm1 = s.residues[rs_picked[1]][0]
                atom0 = s.residues[rs_picked[0]][i]
                atom1 = s.residues[rs_picked[1]][i]
                atom0 = atom0[0:21] + atom_norm0[21:26] + atom0[26:54] + atom_norm0[54:76] + sc[str(i)].rjust(2) + '\n'
                # print(atom0)
                s.residues[rs_picked[0]][i] = atom0
                atom1 = atom1[0:21] + atom_norm1[21:26] + atom1[26:54] + atom_norm1[54:76] + sc[str(i)].rjust(2) + '\n'
                s.residues[rs_picked[1]][i] = atom1
                # print(atom0, atom1, atom_norm0, atom_norm1)
        s.pdb_atoms = residues2atoms(s.residues)
        s.norm_pdb_atoms(lig_ch)
        # print('SSbond successfully cyclized!')
        count+=1
        if count >= max_num:
            return s
    if count ==0:
        return None
    return s

def merge_s(s1,s2,fp):
    pdb_atoms = s1.pdb_atoms + s2.pdb_atoms
    f = open(fp,'w')
    f.writelines(pdb_atoms)
    f.close()
    return fp

if __name__ == '__main__':
    ssdata = read_ssdata('./SSBLIB/')
    pep_s = Structure('./A_1GI9_1.8_fixed_sample_0.pdb',chains = ['a']) 
    rec_s = Structure('./A_1GI9_1.8_fixed_sample_0.pdb',chains = ['b']) 
    cys_terminal(pep_s,[])  # index to be preserved (peptide from 1)
    s_new = build_ssbond(pep_s, ssdata, 2)  # maximum number of SS bonds
    merge_s(rec_s, s_new, './test_ss.pdb')
    
