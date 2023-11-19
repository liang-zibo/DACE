import lightning.pytorch as pl
from torch.utils.data import DataLoader
import argparse
from model import PL_DACE, DACELora, PLTrainer
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from ray.tune import TuneConfig
from utils import *
from plan_utils import *


def prepare_plans(configs):
    statistics_file_path = configs["statistics_path"]

    feature_statistics = load_json(ROOT_DIR + statistics_file_path)
    # add numerical scalers (cite from zero-shot)
    add_numerical_scalers(feature_statistics)

    # op_name to one-hot, using feature_statistics
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)

    # get plans meta
    # plans_meta: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
    plans_meta = process_plans(
        configs, op_name_to_one_hot, plan_parameters, feature_statistics
    )

    return plans_meta


def train(configs):
    # get plans meta
    plans_meta = prepare_plans(configs)
    # create training, test set, based on database_id
    test_database_ids = configs["test_database_ids"]
    if not isinstance(test_database_ids, list):
        test_database_ids = [test_database_ids]
    train_data, test_data = [], []

    for plan_meta in plans_meta:
        if plan_meta[-1] in test_database_ids:
            test_data.append(plan_meta[:-1])
        else:
            train_data.append(plan_meta[:-1])

    # split train_data into train_data and val_data by 9:1
    train_data, val_data = train_test_split(
        train_data, test_size=0.1, random_state=configs["random_seed"]
    )

    # create dataset
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    test_dataset = prepare_dataset(test_data)

    batch_size = configs["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train model
    model = DACELora(
        configs["node_length"],
        configs["hidden_dim"],
        1,
        configs["mlp_activation"],
        configs["transformer_activation"],
        configs["mlp_dropout"],
        configs["transformer_dropout"],
    )

    model = PL_DACE(model)

    wandb_logger = pl.loggers.WandbLogger(project="DACE")

    wandb_logger.log_hyperparams(configs)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(ROOT_DIR, "checkpoints"),
        filename="DACE",
    )
    trainer = PLTrainer(
        accelerator="gpu",
        devices=[0],
        enable_progress_bar=configs["progress_bar"],
        enable_model_summary=configs["progress_bar"],
        max_epochs=configs["max_epoch"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    # trainer = PLTrainer(accelerator="cpu", max_epochs=50, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    # test model
    result = trainer.test(model, dataloaders=test_dataloader)
    return result


def train_with_tune(configs):
    result = train(configs)
    return {"test_database_ids": configs["test_database_ids"], **result}


def train_with_ray(configs):
    ray_train = tune.with_resources(
        train_with_tune,
        {
            "CPU": 1,
            "GPU": 0.2,
        },
    )
    tune_config = TuneConfig(max_concurrent_trials=10)
    experiment_spaces = {
        **configs,
        "test_database_ids": tune.grid_search(list(range(20))),
    }
    tuner = tune.Tuner(
        ray_train,
        param_space=experiment_spaces,
        tune_config=tune_config,
    )

    results = tuner.fit()
    df_results = results.get_dataframe()
    # save results to a csv file
    df_results.to_csv(ROOT_DIR + "results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=123, help="random seed")
    parser.add_argument("--node_length", type=int, default=18, help="node length")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden dimension in transformer"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--pad_length", type=int, default=20, help="pad length")
    parser.add_argument("--max_epoch", type=int, default=50, help="max epoch")
    parser.add_argument(
        "--test_database_ids",
        type=int,
        default=13,
        nargs="+",
        help="test database list",
    )
    parser.add_argument(
        "--loss_weight",
        type=float,
        default=0.5,
        help="loss weight in tree structure-based loss adjustment strategy",
    )
    parser.add_argument(
        "--mlp_activation",
        type=str,
        default="ReLU",
        help="activation function in MLP (can be LeakyReLU, ReLU, GELU))",
    )
    parser.add_argument(
        "--transformer_activation",
        type=str,
        default="gelu",
        help="activation function in transformer (can be relu, gelu)",
    )
    parser.add_argument(
        "--mlp_dropout", type=float, default=0.3, help="dropout rate in MLP"
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.2,
        help="dropout rate in transformer",
    )
    parser.add_argument(
        "--plans_dir",
        type=str,
        default="data/workload1",
        help="plans directory",
    )
    parser.add_argument(
        "--statistics_path",
        type=str,
        default="data/workload1/statistics.json",
        help="statistics file path",
    )
    parser.add_argument(
        "--process_plans", action="store_true", help="process plans to features"
    )
    parser.add_argument(
        "--test_all",
        action="store_true",
        help="whether to test all database (if false, only test the database in test_database_ids)",
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="whether to show progress bar in training",
    )

    args = parser.parse_args()

    # transform args to configs
    configs = vars(args)
    configs["max_runtime"] = 30000

    # set random seed
    set_seed(configs["random_seed"])
    if configs["process_plans"]:
        prepare_plans(configs)
    elif configs["test_all"]:
        train_with_ray(configs)
    else:
        train(configs)
