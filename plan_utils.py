import os
import torch
from torch.nn.functional import pad
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from utils import *
from data_utils import *


# create q-error function: max(pred/target, target/pred)
def q_error(pred, target):
    return torch.max(pred / target, target / pred)


def q_error_np(pred, target):
    return np.max([pred / target, target / pred], axis=0)


def get_op_name_to_one_hot(feature_statistics):
    op_name_to_one_hot = {}
    op_names = feature_statistics["node_types"]["value_dict"]
    op_names_no = len(op_names)
    for i, name in enumerate(op_names.keys()):
        op_name_to_one_hot[name] = np.zeros((1, op_names_no), dtype=np.int32)
        op_name_to_one_hot[name][0][i] = 1
    return op_name_to_one_hot


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v["type"] == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v["center"]
            scaler.scale_ = v["scale"]
            feature_statistics[k]["scaler"] = scaler


def dfs(plan, seq, adjs, parent_node_id, run_times, heights, cur_height):
    cur_node_id = len(seq)
    seq.append(plan)
    heights.append(cur_height)
    run_times.append(plan["Actual Total Time"])
    if parent_node_id != -1:  # not root node
        adjs.append((parent_node_id, cur_node_id))
    if "Plans" in plan:
        for child in plan["Plans"]:
            dfs(child, seq, adjs, cur_node_id, run_times, heights, cur_height + 1)


def get_plan_sequence(plan, pad_length=20):
    """
    plan: a plan read from json file
    pad_length: int, the length of the padded seqs (the number of nodes in the plan)

    return: seq, run_times, adjs, heights, database_id
    seq: List, each element is a node's plan_parameters
    run_times: List, each element is a node's runtime
    adjs: List, each element is a tuple of (parent, child)
    heights: List, each element is a node's height
    database_id: int, the id of the database
    """
    # get all sub-plans' runtime
    seq = []
    run_times = []
    adjs = []  # [(parent, child)]
    heights = []  # the height of each node, root node's height is 0
    dfs(plan["Plan"], seq, adjs, -1, run_times, heights, 0)
    # padding run_times to the same length
    if len(run_times) < pad_length:
        run_times = run_times + [1] * (pad_length - len(run_times))
    return seq, run_times, adjs, heights, plan["database_id"]


def scale_feature(feature_statistics, feature, node):
    if feature_statistics[feature]["type"] == str(FeatureType.numeric):
        scaler = feature_statistics[feature]["scaler"]
        return scaler.transform(np.array([node[feature]]).reshape(-1, 1))
    else:
        return feature_statistics[feature]["value_dict"][node["Node Type"]]


def generate_seqs_encoding(
    seq, op_name_to_one_hot, plan_parameters, feature_statistics
):
    seq_encoding = []
    for node in seq:
        # add op_name encoding
        op_name = node[plan_parameters[0]]
        op_encoding = op_name_to_one_hot[op_name]
        seq_encoding.append(op_encoding)
        # add other features, and scale them
        for feature in plan_parameters[1:]:
            feature_encoding = scale_feature(feature_statistics, feature, node)
            seq_encoding.append(feature_encoding)
    seq_encoding = np.concatenate(seq_encoding, axis=1)

    return seq_encoding


def pad_sequence(seq_encoding, padding_value=0, node_length=18, max_length=20):
    """
    pad seqs to the same length, and transform seqs to a tensor
    """
    # seqs: list of seqs (seq's shape: (1, feature_no)))
    # padding_value: padding value
    # return: padded seqs, seqs_length
    seq_length = seq_encoding.shape[1]
    seq_padded = pad(
        torch.from_numpy(seq_encoding),
        (0, max_length * node_length - seq_encoding.shape[1]),
        value=padding_value,
    )
    seq_padded = seq_padded.to(dtype=torch.float32)
    return seq_padded, seq_length


# get attention mask
def get_attention_mask(adj, seq_length, pad_length, node_length):
    # adjs: List, each element is a tuple of (parent, child)
    # seqs_length: List, each element is the length of a seq
    # pad_length: int, the length of the padded seqs
    # return: attention mask
    seq_length = int(seq_length / node_length)

    attention_mask_seq = np.ones((pad_length, pad_length))
    for a in adj:
        attention_mask_seq[a[0], a[1]] = 0

    # based on the reachability of the graph, set the attention mask
    for i in range(seq_length):
        for j in range(seq_length):
            if attention_mask_seq[i, j] == 0:
                for k in range(seq_length):
                    if attention_mask_seq[j, k] == 0:
                        attention_mask_seq[i, k] = 0

    # node can reach itself
    for i in range(pad_length):
        attention_mask_seq[i, i] = 0

    # to tensor
    attention_mask_seq = torch.tensor(attention_mask_seq, dtype=torch.bool)
    return attention_mask_seq


def get_loss_mask(seq_length, pad_length, node_length, height, loss_weight=0.5):
    seq_length = int(seq_length / node_length)
    loss_mask = np.zeros((pad_length))
    loss_mask[:seq_length] = np.power(loss_weight, np.array(height))
    loss_mask = torch.from_numpy(loss_mask).float()
    return loss_mask


# get a plan's encoding
def get_plan_encoding(
    plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics
):
    """
    plan: a plan read from json file
    pad_length: int, the length of the padded seqs (the number of nodes in the plan)
    """
    seq, run_times, adjs, heights, database_id = get_plan_sequence(
        plan, configs["pad_length"]
    )
    run_times = np.array(run_times).astype(np.float32) / configs["max_runtime"] + 1e-7
    run_times = torch.from_numpy(run_times)
    seq_encoding = generate_seqs_encoding(
        seq, op_name_to_one_hot, plan_parameters, feature_statistics
    )

    # pad seq_encoding
    seq_encoding, seq_length = pad_sequence(
        seq_encoding,
        padding_value=0,
        node_length=configs["node_length"],
        max_length=configs["pad_length"],
    )

    # get attention mask
    attention_mask = get_attention_mask(
        adjs, seq_length, configs["pad_length"], configs["node_length"]
    )

    # get loss mask
    loss_mask = get_loss_mask(
        seq_length,
        configs["pad_length"],
        configs["node_length"],
        heights,
        configs["loss_weight"],
    )

    return seq_encoding, run_times, attention_mask, loss_mask, database_id


def process_plans(
    configs,
    op_name_to_one_hot,
    plan_parameters,
    feature_statistics,
    pre_process_path="data/workload1/plans_meta.pkl",
):
    if os.path.exists(os.path.join(ROOT_DIR, pre_process_path)):
        plans_meta = load_pickle(os.path.join(ROOT_DIR, pre_process_path))
        return plans_meta

    # read plans
    plans = read_workload_runs(
        os.path.join(configs["plans_dir"]), db_names=workloads, verbose=True
    )

    print("generating encoding...")
    plans_meta = []
    for plan in tqdm(plans):
        # get plan encoding, plan_meta: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
        plan_mata = get_plan_encoding(
            plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        )
        plans_meta.append(plan_mata)

    save_pickle(plans_meta, os.path.join(ROOT_DIR, pre_process_path))
    return plans_meta


def prepare_dataset(data):
    """
    data: List, each element is a tuple of (seq, run_time, attention_mask, loss_mask)
    """
    seqs, run_times, attention_masks, loss_masks = [], [], [], []
    for seq, run_time, attention_mask, loss_mask in data:
        seqs.append(seq)
        run_times.append(run_time)
        attention_masks.append(attention_mask)
        loss_masks.append(loss_mask)
    seqs = torch.stack(seqs)
    run_times = torch.stack(run_times)
    attention_masks = torch.stack(attention_masks)
    loss_masks = torch.stack(loss_masks)

    dataset = DACEDataset(seqs, attention_masks, loss_masks, run_times)
    return dataset
