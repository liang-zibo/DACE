import json
from enum import Enum
import torch
import numpy as np
import os
import pickle


ROOT_DIR = "/data1/liangzibo/dace/"

plan_parameters = [
    "Node Type",
    "Total Cost",
    "Plan Rows",
]

workloads = [
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "consumer",
    "credit",
    "employee",
    "fhnk",
    "financial",
    "geneea",
    "genome",
    "hepatitis",
    "imdb_full",
    "movielens",
    "seznam",
    "ssb",
    "tournament",
    "tpc_h",
    "walmart",
]


def load_json(path):
    with open(path) as json_file:
        json_obj = json.load(json_file)
    return json_obj


class FeatureType(Enum):
    numeric = "numeric"
    categorical = "categorical"

    def __str__(self):
        return self.value


# set all random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def getModelSize(model):
    param_size = 0
    param_sum = 0
    lora_size = 0
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_size += param.nelement() * param.element_size()
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print("Param: {:.3f}MB".format(param_size / 1024 / 1024))
    print("Buffer: {:.3f}MB".format(buffer_size / 1024 / 1024))
    print("Lora: {:.3f}MB".format(lora_size / 1024 / 1024))
    print("Total Size: {:.3f}MB".format(all_size))


def get_workload_paths(workload):
    workload_file_dir = "data/workload1"
    workload_file_dir = os.join(ROOT_DIR, workload)
    path_list = os.listdir(workload_file_dir)
    for dbtype in path_list:
        if os.path.isfile(os.path.join(workload_file_dir, dbtype)):
            path_list.remove(dbtype)
    return path_list


def read_workload_runs(workload_dir, db_names, verbose=False):
    # reads several workload runs
    plans = []

    for i, source in enumerate(db_names):
        try:
            workload_path = os.path.join(workload_dir, source + "_filted.json")
            run = load_json(workload_path)
        except json.JSONDecodeError:
            raise ValueError(f"Error reading {source}")

        db_count = 0
        for plan_id, plan in enumerate(run):
            plan["database_id"] = i
            plan["plan_id"] = plan_id
            plans.append(plan)
            db_count += 1
        if verbose:
            print("Database {:s} has {:d} plans.".format(source, db_count))
    print("Total number of plans: {:d}".format(len(plans)))

    return plans


def print_qerrors(qerrors):
    # qerrors: tensor or numpy array
    # print 50th, 90th, 95th, 99th quantile, min and max errors
    if isinstance(qerrors, torch.Tensor):
        qerrors = qerrors.detach().cpu().numpy()
    print("50th quantile: ", np.quantile(qerrors, 0.5))
    print("90th quantile: ", np.quantile(qerrors, 0.9))
    print("95th quantile: ", np.quantile(qerrors, 0.95))
    print("99th quantile: ", np.quantile(qerrors, 0.99))
    print("max: ", np.max(qerrors))
    print("mean: ", np.mean(qerrors))
