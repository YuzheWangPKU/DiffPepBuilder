# Target-Specific De Novo Peptide Binder Design with DiffPepBuilder

This is the official repository for the paper [Target-Specific De Novo Peptide Binder Design with DiffPepBuilder](https://arxiv.org/abs/2405.00128).

![plot](dpb_model.jpg)

For any questions, please open an [issue](https://github.com/YuzheWangPKU/DiffPepBuilder/issues) or contact wangyuzhe_ccme@pku.edu.cn for more information.

## Quick Start
We provide a colaboratory notebook to demonstrate the usage of DiffPepBuilder (*in progress*). Please click the following link to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YuzheWangPKU/DiffPepBuilder/blob/main/examples/DiffPepBuilder_demo.ipynb)

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

Before running inference, please unzip the SSBLIB data in the `SSbuilder` directory:

```bash
cd SSbuilder
tar -xvf SSBLIB.tar.gz
```

The post-processing procedure requires [Rosetta](https://rosettacommons.org/software/) to be installed. Please download the latest version of Rosetta from the [official website](https://rosettacommons.org/download/) and follow the [installation instructions](https://docs.rosettacommons.org/docs/latest/getting_started/Getting-Started).

## Inference
To *de novo* generate peptide binders for a given target protein, please first download the model weights into `experiments/checkpoints/` from [Zenodo](https://zenodo.org/records/12794439). You can use the following command to download the model weights:

```bash
wget https://zenodo.org/records/12794439/files/diffpepbuilder_v1.pth
mv diffpepbuilder_v1.pth experiments/checkpoints/
```

We provide an example of the target ALK1 (Activin Receptor-like Kinase 1, PDB ID: [6SF1](https://www.rcsb.org/structure/6SF1)) to demonstrate the procedures of generating peptide binders. Please note that the following pipeline can also be used to generate peptide binders for multiple targets simultaneously. The hotspots or binding motif of the target protein can be specified in JSON format, as showcased by the example file `examples/receptor_data/de_novo_cases.json`. To preprocess the receptor, run the `experiments/process_receptor.py` script:

```bash
python experiments/process_receptor.py --pdb_dir examples/receptor_data --write_dir data/receptor_data --peptide_info_path examples/receptor_data/de_novo_cases.json
```

This script will generate the receptor data in the `data/receptor_data` directory. To generate peptide binders for the target protein, please specify the root directory of DiffPepBuilder repository and then run the `experiments/run_inference.py` script (modify the `nproc-per-node` flag accordingly based on the number of GPUs available):

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
torchrun --nproc-per-node=8 experiments/run_inference.py data.val_csv_path=data/receptor_data/metadata_test.csv
```

The config file `configs/inference.yaml` contains the hyperparameters for the inference process. Below is a brief explanation of the key hyperparameters:

| Parameter            | Description                                                                         | Default Value |
|----------------------|-------------------------------------------------------------------------------------|---------------|
| **use_ddp**          | Indicates whether Distributed Data Parallel (DDP) training is used                  | True          |
| **use_gpu**          | Specifies whether to use GPU for computation                                        | True          |
| **num_gpus**         | Number of GPUs to use for computation                                               | 8             |
| **num_t**            | Number of denoising steps                                                           | 200           |
| **noise_scale**      | Scaling factor for noise, analogous to sampling temperature                         | 1.0           |
| **samples_per_length** | Number of peptide backbone samples per sequence length                              | 8           |
| **min_length**       | Minimum sequence length to sample                                                   | 8             |
| **max_length**       | Maximum sequence length to sample                                                   | 30            |
| **seq_temperature**  | Sampling temperature of the residue types                                                                | 0.1           |
| **build_ss_bond**    | Indicates whether to build disulfide bonds                                          | True          |
| **max_ss_bond**      | Maximum number of disulfide (SS) bonds to build                                     | 2             |

You can modify these hyperparameters to customize the inference process. For more details on the hyperparameters, please refer to our [paper](https://arxiv.org/abs/2405.00128). 

After running the inference script, the generated peptide binders will be saved in the `tests/inference/`. To run the side chain assembly and energy minimization using [Rosetta](https://rosettacommons.org/software/), please run the following script subsequently:

```bash
export BASE_PATH="your/path/to/DiffPepBuilder"
python experiments/run_redock.py --in_path tests/inference --ori_path examples/receptor_data --interface_analyzer_path your/path/to/rosetta/main/source/bin/rosetta_scripts.static.linuxgccrelease
```

Modify the `interface_analyzer_path` flag to the path of the Rosetta `interface_analyzer` executable. The script will generate the final peptide binders in the `tests/inference/.../pdbs_redock/` directory and calculate the binding ddG values of the generated peptide binders. The results will be summarized in the `tests/inference/redock_results.csv` file.

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

The config file `configs/base.yaml` contains the hyperparameters for the training process. You can modify these hyperparameters to customize the training process. Checkpoints will be saved every 10,000 steps after validation in the `tests/ckpt/` directory by default. Training logs will be saved every 2,500 steps.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
Please cite the following paper if you use this code in your research:
```
@misc{wang2024targetspecificnovopeptidebinder,
      title={Target-Specific De Novo Peptide Binder Design with DiffPepBuilder}, 
      author={Fanhao Wang and Yuzhe Wang and Laiyi Feng and Changsheng Zhang and Luhua Lai},
      year={2024},
      eprint={2405.00128},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2405.00128}, 
}
```

## Acknowledgments
We would like to thank the authors of [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) and [OpenFold](https://github.com/aqlaboratory/openfold), whose codebases we used as references for our implementation.
