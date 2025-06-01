"""
Script for postprocessing procedures.
"""
import pyrootutils

# See: https://github.com/ashleve/pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True
)

import os
import argparse
from datetime import datetime
from analysis.metrics import postprocess_metric_parallel

now = datetime.now()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run postprocessing procedures."
    )

    parser.add_argument(
        "--in_pdbs",
        type=str,
        required=True,
        help="Path to the directory containing the test PDB files."
    )

    parser.add_argument(
        "--ori_pdbs",
        type=str,
        default=f"{root}/data/receptor_data/receptor_raw_data",
        help="Path to the directory containing native pdb complexes."
    )
    
    parser.add_argument(
        "--nproc",
        type=int,
        default=os.cpu_count()-1,
        help="Number of processing cores to use."
    )

    parser.add_argument(
        "--postprocess_xml_path",
        type=str,
        default=f"{root}/analysis/interface_analyze.xml",
        help="Path to the XML file containing the postprocessing protocol."
    )

    parser.add_argument(
        "--amber_relax",
        action='store_true',
        help="Relax the structures using Amber force field."
    )

    parser.add_argument(
        "--rosetta_relax",
        action='store_true',
        help="Relax the structures using Rosetta FastRelax."
    )

    return parser


def main(args):
    test_files = []
    for root, dirs, files in os.walk(args.in_pdbs):
        for file in files:
            if file.endswith('.pdb'):
                test_files.append(os.path.join(root, file))
    
    print(f'Start postprocessing of peptides...')
    postprocess_metric_parallel(
        files=test_files,
        ori_dir=args.ori_pdbs,
        lig_chs=['A' for _ in test_files],
        xml=args.postprocess_xml_path,
        out_path=os.path.join(args.in_pdbs, 'postprocess_results.csv'),
        amber_relax=args.amber_relax,
        rosetta_relax=args.rosetta_relax
    )
    print(f'Finished postprocessing of peptides.')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

