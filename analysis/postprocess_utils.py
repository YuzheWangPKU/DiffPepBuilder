# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for minimization."""

import io
from Bio.PDB import *
import numpy as np
from typing import List
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.PDBExceptions import PDBIOException
from pdbfixer import PDBFixer
import os
import openmm.app as app
import json
from typing import List
import pandas as pd

_TER_FORMAT_STRING = (
    "TER   %5i      %3s %c%4i%c                                                      \n"
)

_select = Select()


class PCLFixer(PDBFixer):
    def __init__(self, file_string=None):
        # Check to make sure only one option has been specified.
        self.source = None
        file = io.StringIO(file_string)
        self._initializeFromPDB(file)

        # Check the structure has some atoms in it.
        atoms = list(self.topology.atoms())
        if len(atoms) == 0:
            raise Exception("Structure contains no atoms.")

        # Load the templates.
        self.templates = {}
        templatesPath = os.path.join(os.path.dirname(__file__), "templates")
        for file in os.listdir(templatesPath):
            templatePdb = app.PDBFile(os.path.join(templatesPath, file))
            name = next(templatePdb.topology.residues()).name
            self.templates[name] = templatePdb


class PCLIO(PDBIO):

    def set_structure_multiple(self, pdb_object: List[object]):
        """Check what the user is providing and build a structure."""
        # The idea here is to build missing upstream components of
        # the SMCRA object representation. E.g., if the user provides
        # a Residue, build Structure/Model/Chain.

        if pdb_object[0].level == "S":
            structure = pdb_object
        else:  # Not a Structure
            sb = StructureBuilder()
            sb.init_structure("pdb")
            sb.init_seg(" ")

            if pdb_object[0].level == "M":
                for i in pdb_object:
                    sb.structure.add(i.copy())
                self.structure = sb.structure
            else:  # Not a Model
                sb.init_model(0)

                if pdb_object[0].level == "C":
                    for i in pdb_object:
                        sb.structure[0].add(i.copy())
                else:  # Not a Chain
                    chain_id = "A"  # default
                    sb.init_chain(chain_id)

                    if pdb_object.level == "R":  # Residue
                        # Residue extracted from a larger structure?
                        if pdb_object[0].parent is not None:
                            og_chain_id = pdb_object[0].parent.id
                            sb.structure[0][chain_id].id = og_chain_id
                            chain_id = og_chain_id

                        for i in pdb_object:
                            sb.structure[0][chain_id].add(i.copy())
            # Return structure
            structure = sb.structure
        self.structure = structure

    def pdb_string(self, select=_select, preserve_atom_numbering=False, write_end=True):
        string = []
        get_atom_line = self._get_atom_line
        # multiple models?
        if len(self.structure) > 1 or self.use_model_flag:
            model_flag = 1
        else:
            model_flag = 0
        for model in self.structure.get_list():
            if not select.accept_model(model):
                continue
            # necessary for ENDMDL
            # do not write ENDMDL if no residues were written
            # for this model
            model_residues_written = 0
            if not preserve_atom_numbering:
                atom_number = 1
            if model_flag:
                string.append(f"MODEL      {model.serial_num}\n")
            for chain in model.get_list():
                if not select.accept_chain(chain):
                    continue
                chain_id = chain.id
                if len(chain_id) > 1:
                    e = f"Chain id ('{chain_id}') exceeds PDB format limit."
                    raise PDBIOException(e)
                # necessary for TER
                # do not write TER if no residues were written
                # for this chain
                chain_residues_written = 0
                for residue in chain.get_unpacked_list():
                    if not select.accept_residue(residue):
                        continue
                    hetfield, resseq, icode = residue.id
                    resname = residue.resname
                    segid = residue.segid
                    resid = residue.id[1]
                    if resid > 9999:
                        e = f"Residue number ('{resid}') exceeds PDB format limit."
                        raise PDBIOException(e)
                    for atom in residue.get_unpacked_list():
                        if not select.accept_atom(atom):
                            continue
                        chain_residues_written = 1
                        model_residues_written = 1
                        if preserve_atom_numbering:
                            atom_number = atom.serial_number
                        try:
                            s = get_atom_line(
                                atom,
                                hetfield,
                                segid,
                                atom_number,
                                resname,
                                resseq,
                                icode,
                                chain_id,
                            )
                        except Exception as err:
                            # catch and re-raise with more information
                            raise PDBIOException(
                                f"Error when writing atom {atom.full_id}"
                            ) from err
                        else:
                            string.append(s)
                            # inconsequential if preserve_atom_numbering is True
                            atom_number += 1
                if chain_residues_written:
                    string.append(
                        _TER_FORMAT_STRING
                        % (atom_number, resname, chain_id, resseq, icode)
                    )
            if model_flag and model_residues_written:
                string.append("ENDMDL\n")
        if write_end:
            string.append("END   \n")
        return "".join(string)


def summarize_statistics(score_files: List[str], keys=["ddg_norepack", "rmsd"]):
    """
    Dataframe of scores by default, first column is affinity or ddG, second represents rmsd before and after postprocessing.
    """
    scores = []
    indexes = []
    for score_file in score_files:
        json_file = score_file.replace(".sc", ".json")
        jsf = open(json_file, "w")
        sf = open(score_file)
        lines = [i for i in sf.readlines()]
        sf.close()
        for idx, line in enumerate(lines):
            lines[idx] = f'"{idx}" : ' + line.strip() + ",\n"
        lines = [i for i in lines if ("nan" not in i) and (len(i) >= 100)]
        lines[0] = "{" + lines[0]
        lines[-1] = lines[-1].strip()[:-1] + "}\n"
        jsf.writelines(lines)
        jsf.close()
        with open(json_file) as js:
            _scores = json.load(js)
        for i in _scores:
            try:
                scores.append([_scores[i][j] for j in keys])
                indexes.append(
                    os.path.join(os.path.dirname(score_file), _scores[i]["decoy"])
                )
            except:
                continue
    data = {"keys": keys, "scores": scores, "paths": indexes}
    return pd.DataFrame(
        data=data["scores"], index=data["paths"], columns=data["keys"]
    )
