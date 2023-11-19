# DACE

## Overview
DACE: A Database-Agnostic Cost Estimator.

## Getting Started

### Prerequisites
- Python 3.9.18
- Required Python packages (see `requirements.txt`)

### Installation
Clone the repository and install the dependencies:
```bash
git clone git@github.com:liang-zibo/DACE.git
cd dace
pip install -r requirements.txt
```

### Download DACE Data
Before running the code, please download the data from this
[data repository](https://figshare.com/s/58a0e03829db15bef555) and put them in the data folder.

## Usage
### Modify ROOT_DIR
Modify ROOT_DIR in utils.py to your own path.

### Filtering Plans and Gathering Statistics
To filter out plans and gather statistical data, run:
```bash
python setup.py --filter_plans --get_statistic
```

### Get plan encodings
To get plan encodings, run:
```bash
python run.py --process_plans
```

### Testing All Databases
To sequentially use each database as a test set while treating the remaining databases as a training set, execute:
```bash
python run.py --test_all
```

### Testing on IMDB Dataset
To test and evaluate DACE's performance on the IMDB dataset (dataset ID: 13), without including any knowledge from the IMDB dataset in the training set, use:
```bash
python run.py --test_database_ids 13
cd data
mv DACE.ckpt DACE_imdb.ckpt
```

### Direct Testing on Workloads
To directly test DACE as a pre-trained estimator on job-light, scale, and synthetic workloads:
```bash
python run_tuning.py
```

### Tuning and Testing on Workloads
For fine-tuning and testing DACE as a pre-trained estimator on job-light, scale, and synthetic workloads:
```bash
python run_tuning.py --tune
```

## Contact

If you have any questions about the code, please email [zibo_liang@outlook.com](mailto:zibo_liang@outlook.com)