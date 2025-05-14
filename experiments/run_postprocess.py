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
from analysis.metrics import redock_metric_parallel

now = datetime.now()


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run postprocessing procedures."
    )

    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to the directory containing the test PDB files."
    )

    parser.add_argument(
        "--ori_path",
        type=str,
        default=f"{root}/data/receptor_data/receptor_raw_data",
        help="Path to the directory containing native pdb complexes."
    )

    parser.add_argument(
        "--redock_app",
        type=str,
        default="interface_analyzer",
        choices=["ADCP", "flexpepdock", "interface_analyzer"],
        help="Specify the redocking application to use (ADCP, flexpepdock, or interface_analyzer). Defaults to interface_analyzer."
    )
    
    parser.add_argument(
        "--nproc",
        type=int,
        default=os.cpu_count()-1,
        help="Number of processing cores to use."
    )

    parser.add_argument(
        "--redock_xml_path",
        type=str,
        default=f"{root}/analysis/interface_analyze.xml",
        help="Path to the XML file containing the redocking protocol."
    )

    parser.add_argument(
        "--flexpepdock_path",
        type=str,
        default="rosetta/rosetta_bin_linux_2021.16.61629_bundle/main/source/bin/FlexPepDocking.default.linuxgccrelease",
        help="Path to the FlexPepDock binary."
    )

    parser.add_argument(
        "--interface_analyzer_path",
        type=str,
        default="rosetta/rosetta_bin_linux_2021.16.61629_bundle/main/source/bin/rosetta_scripts.static.linuxgccrelease",
        help="Path to the interface analyzer binary."
    )

    return parser


def main(args):
    test_files = []
    for root, dirs, files in os.walk(args.in_path):
        for file in files:
            if file.endswith('.pdb'):
                test_files.append(os.path.join(root, file))
    app_paths = {
        "flexpepdock": args.flexpepdock_path,
        "interface_analyzer": args.interface_analyzer_path,
        "ADCP": None
    }
    redock_app_path = app_paths[args.redock_app]
    
    print(f'Start postprocessing of peptides with {args.redock_app}...')
    redock_metric_parallel(
        files=test_files,
        ori_path=args.ori_path,
        nproc=args.nproc,
        app=args.redock_app,
        app_path=redock_app_path,
        lig_chs=['A' for _ in test_files],
        denovo=True,
        xml=args.redock_xml_path,
        out_path=os.path.join(args.in_path, 'redock_results.csv')
    )
    print(f'Finished postprocessing of peptides.')


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

