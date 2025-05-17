# Target-Specific *De Novo* Peptide Binder Design with DiffPepBuilder

This is the official repository for the [DiffPepBuilder](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00975) and DiffPepDock tools.

![plot](dpb_model.jpg)

For any questions, please open an [issue](https://github.com/YuzheWangPKU/DiffPepBuilder/issues) or contact wangyuzhe_ccme@pku.edu.cn for more information.

## News
* **[2025/5/14]** We extend our approach to protein–peptide docking through a derivative tool, DiffPepDock. The initial implementation and [model weights](https://zenodo.org/records/15398020) for the docking functionality have been publicly released.
* **[2024/9/12]** Our research article is now published in **JCIM**! Dive into the details by checking out the full paper [here](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00975) or on [ArXiv](https://arxiv.org/abs/2405.00128).
* **[2024/9/11]** We released the [PepPC-F](datasets/PepPC-F_dataset.csv) and [PepPC](datasets/PepPC_dataset.csv) datasets for DiffPepBuilder on [Zenodo](https://zenodo.org/records/13744959). The training protocol has also been released. Please refer to the [Training](#training) section for more details.
* **[2024/7/22]** We released the initial code, [model weights](https://zenodo.org/records/12794439), and a [Colab demo](https://colab.research.google.com/github/YuzheWangPKU/DiffPepBuilder/blob/main/examples/DiffPepBuilder_demo.ipynb) for DiffPepBuilder.


## Quick Start
We provide a Google Colab notebook to facilitate the use of DiffPepBuilder. Please click the following link to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuzheWangPKU/DiffPepBuilder/blob/main/examples/DiffPepBuilder_demo.ipynb)

Similarly, a Colab notebook demonstrating the functionality of DiffPepDock is available at:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuzheWangPKU/DiffPepBuilder/blob/main/examples/DiffPepDock_demo.ipynb)

## Installation
We recommend using a conda environment to install the required packages. Please clone this repository and navigate to the root directory:
    
```bash
git clone https://github.com/YuzheWangPKU/DiffPepBuilder.git
cd DiffPepBuilder
```

Then run the following commands to create a new conda environment and install the required packages:

```bash
conda env create -f environment.yml
conda activate diffpepbuilder
```

Before running *de novo* design protocols, please unzip the SSBLIB data in the `SSbuilder` directory:

```bash
cd SSbuilder
tar -xvf SSBLIB.tar.gz
```

The post-processing procedure requires [Rosetta](https://rosettacommons.org/software/) to be installed. Please download the latest version of Rosetta from the [official website](https://rosettacommons.org/download/) and follow the [installation instructions](https://docs.rosettacommons.org/docs/latest/getting_started/Getting-Started).

## *De Novo* Design
To *de novo* generate peptide binders for a given target protein, please first download the model weights into `experiments/checkpoints/` from [Zenodo](https://zenodo.org/records/12794439). You can use the following command to download the model weights:

```bash
wget https://zenodo.org/records/12794439/files/diffpepbuilder_v1.pth
mv diffpepbuilder_v1.pth experiments/checkpoints/
```

We provide an example of the target ALK1 (Activin Receptor-like Kinase 1, PDB ID: [6SF1](https://www.rcsb.org/structure/6SF1)) to demonstrate the procedures of generating peptide binders. Please note that the following pipeline can also be used to generate peptide binders for multiple targets simultaneously. The hotspots or binding motif of the target protein can be specified in JSON format, as showcased by the example file `examples/receptor_data/de_novo_cases.json`. To preprocess the receptor, run the `experiments/process_receptor.py` script:

```bash
python experiments/process_receptor.py --pdb_dir examples/receptor_data --write_dir data/receptor_data --receptor_info_path examples/receptor_data/de_novo_cases.json
```

This script will generate the receptor data in the `data/receptor_data` directory. To generate peptide binders for the target protein, please specify the root directory of DiffPepBuilder repository and then run the `experiments/run_inference.py` script (modify the `nproc-per-node` flag accordingly based on the number of GPUs available):

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
torchrun --nproc-per-node=8 experiments/run_inference.py data.val_csv_path=data/receptor_data/metadata_test.csv
```

The config file `config/inference.yaml` contains the hyperparameters for the inference process. Below is a brief explanation of the key hyperparameters:

| Parameter            | Description                                                                         | Default Value |
|----------------------|-------------------------------------------------------------------------------------|---------------|
| **use_ddp**          | Indicates whether Distributed Data Parallel (DDP) training is used                  | True          |
| **use_gpu**          | Specifies whether to use GPU for computation                                        | True          |
| **num_gpus**         | Number of GPUs to use for computation                                               | 8             |
| **num_t**            | Number of denoising steps                                                           | 200           |
| **noise_scale**      | Scaling factor for noise, analogous to sampling temperature                         | 1.0           |
| **samples_per_length** | Number of peptide backbone samples per sequence length                            | 8             |
| **min_length**       | Minimum sequence length to sample                                                   | 8             |
| **max_length**       | Maximum sequence length to sample                                                   | 30            |
| **seq_temperature**  | Sampling temperature of the residue types                                           | 0.1           |
| **build_ss_bond**    | Indicates whether to build disulfide bonds                                          | True          |
| **max_ss_bond**      | Maximum number of disulfide (SS) bonds to build                                     | 2             |

You can modify these hyperparameters to customize the inference process. For more details on the hyperparameters, please refer to our [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00975).

After running the inference script, the generated peptide binders will be saved in the `runs/inference/`. To run the side chain assembly and energy minimization using [Rosetta](https://rosettacommons.org/software/), please run the following script subsequently:

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
python experiments/run_postprocess.py --in_path runs/inference --ori_path examples/receptor_data --interface_analyzer_path your/path/to/rosetta/main/source/bin/rosetta_scripts.static.linuxgccrelease
```

Modify the `interface_analyzer_path` flag to the path of the Rosetta `interface_analyzer` executable. The script will generate the final peptide binders in the `runs/inference/.../pdbs_redock/` directory and calculate the binding ddG values of the generated peptide binders. The results will be summarized in the `runs/inference/redock_results.csv` file.

## Docking
To run peptide docking protocols for a given target protein, please first download the model weights into `experiments/checkpoints/` from [Zenodo](https://zenodo.org/records/15398020). You can use the following command to download the model weights:

```bash
wget https://zenodo.org/records/15398020/files/diffpepdock_v1.pth
mv diffpepdock_v1.pth experiments/checkpoints/
```

DiffPepDock provides user-friendly, automated scripts for docking batches of peptide sequences to a specific target protein. Here we provide an example of the redocking task of the substrate-binding protein YejA in complex with its native peptide fragment (PDB ID: [7Z6F](https://www.rcsb.org/structure/7Z6F)) to demonstrate the procedures of docking process. You may modify the file `examples/docking_data/peptide_seq.fasta` include custom peptide sequences for docking. Prior binding information including reference ligands and binding motifs can be specified in JSON format, as demonstrated in `examples/docking_data/docking_cases.json`, using the same syntax as in *de novo* design.

To preprocess the target and the peptide sequences, run the `experiments/process_batch_dock.py` script:

```bash
python experiments/process_batch_dock.py --pdb_dir examples/docking_data --write_dir data/docking_data --receptor_info_path examples/docking_data/docking_cases.json --peptide_seq_path examples/docking_data/peptide_seq.fasta
```

The preprocessed data will be placed in the `data/docking_data` directory. To perform protein-peptide docking, please specify the root directory of DiffPepBuilder repository and then run the `experiments/run_docking.py` script (please modify the `nproc-per-node` flag accordingly based on the number of GPUs available):

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
torchrun --nproc-per-node=8 experiments/run_docking.py data.val_csv_path=data/docking_data/metadata_test.csv
```

The config file `config/docking.yaml` contains the hyperparameters for the docking process. You can modify these hyperparameters to customize the docking process.

After running the inference script, the generated protein-peptide complexes will be saved in the `runs/docking/`. To run the side chain assembly using [Rosetta](https://rosettacommons.org/software/), please run the following script subsequently:

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
python experiments/run_postprocess.py --in_path runs/docking --ori_path examples/docking_data --interface_analyzer_path your/path/to/rosetta/main/source/bin/rosetta_scripts.static.linuxgccrelease
```

Modify the `interface_analyzer_path` flag to the path of the Rosetta `interface_analyzer` executable. The script will generate the final protein-peptide complexes in the `runs/docking/.../pdbs_redock/` directory.

## Training
To train the DiffPepBuilder model from scratch, please download the training data from [Zenodo](https://zenodo.org/records/13744959) and unzip the data in the `data/` directory:

```bash
wget https://zenodo.org/records/13744959/files/PepPC-F_raw_data.tar.gz
mkdir data/PepPC-F_raw_data
tar -xvf PepPC-F_raw_data.tar.gz --strip-components=1 -C data/PepPC-F_raw_data
```

To preprocess the training data, run the `experiments/process_dataset.py` script:

```bash
python experiments/process_dataset.py --pdb_dir data/PepPC-F_raw_data --write_dir data/complex_dataset
```

This script will generate the training data in the `data/complex_dataset` directory. You can add `max_batch_size` flag to specify the maximum batch size for ESM embedding to avoid out-of-memory errors. Then split the data into training and validation sets:

```bash
python experiments/split_dataset.py --input_path data/complex_dataset/metadata.csv --output_path data/complex_dataset --num_val 200
```

You can modify the `num_val` flag to specify the number of validation samples. To train the DiffPepBuilder model, please specify the root directory of the DiffPepBuilder repository and then run the `experiments/train.py` script (modify the `nproc-per-node` flag accordingly based on the number of GPUs available):

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
torchrun --nproc-per-node=8 experiments/train.py
```

The config file `config/base.yaml` contains the hyperparameters for the training process. You can modify these hyperparameters to customize the training process. Checkpoints will be saved every 10,000 steps after validation in the `runs/ckpt/` directory by default. Training logs will be saved every 2,500 steps. 

To run subsequent finetuning for docking tasks, please prepare the PepPC dataset, update the config file `config/finetune.yaml` accordingly, and run the following command:

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
torchrun --nproc-per-node=8 experiments/train.py --config-name=finetune
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
Please cite the following paper if you use this code in your research:
```
@article{wang2024target,
  title={Target-Specific De Novo Peptide Binder Design with DiffPepBuilder},
  author={Wang, Fanhao and Wang, Yuzhe and Feng, Laiyi and Zhang, Changsheng and Lai, Luhua},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={24},
  pages={9135-9149},
  year={2024},
  publisher={ACS Publications},
  doi = {10.1021/acs.jcim.4c00975}
}
```

## Acknowledgments
We would like to thank the authors of [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) and [OpenFold](https://github.com/aqlaboratory/openfold), whose codebases we used as references for our implementation.
