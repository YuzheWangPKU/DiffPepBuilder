# PepPC-F and PepPC Datasets for DiffPepBuilder

This repository provides information on PepPC-F and PepPC datasets for the paper [Target-Specific De Novo Peptide Binder Design with DiffPepBuilder](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00975).

![plot](datasets.jpg)

For any questions, please open an [issue](https://github.com/YuzheWangPKU/DiffPepBuilder/issues) or contact wangyuzhe_ccme@pku.edu.cn for more information.

## PepPC-F Dataset

The contents of PepPC-F dataset (synthetic protein-protein fragment complex dataset) are provided in the file [PepPC-F_dataset.csv](PepPC-F_dataset.csv). To obtain the raw data from [Zenodo](https://zenodo.org/records/13744959), you can use the following command:

```bash
wget https://zenodo.org/records/13744959/files/PepPC-F_raw_data.tar.gz
```

## PepPC Dataset

The contents of PepPC dataset (natural protein-peptide complex dataset) are provided in the file [PepPC_dataset.csv](PepPC_dataset.csv). To obtain the raw data from [Zenodo](https://zenodo.org/records/13744959), you can use the following command:

```bash
wget https://zenodo.org/records/13744959/files/PepPC_raw_data.tar.gz
```

For fine-tuning and evaluation of the DiffPepDock model on protein–peptide docking tasks, the PepPC dataset is split based on deposition date. The training and validation sets, provided in [PepPC_before_202201.csv](PepPC_before_202201.csv), include 3,619 complexes deposited before January 1, 2022. The test set, provided in [PepPC_after_202201.csv](PepPC_after_202201.csv), comprises 205 complexes deposited after this date.