import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import loralib as lora
from utils import *
from plan_utils import *
from model import DACELora, PL_DACE, PLTrainer


# tune
def tune_DACE(configs, model):
    train_data = load_json(os.path.join(ROOT_DIR, "data/workload2/mscn_plans.json"))
    statistics_file_path = configs["statistics_path"]
    feature_statistics = load_json(ROOT_DIR + statistics_file_path)
    # add numerical scalers (cite from zero-shot)
    add_numerical_scalers(feature_statistics)

    # op_name to one-hot, using feature_statistics
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    train_plans = []
    for plan in train_data:
        run_time = plan["plan"][0][0][0]["Plan"]["Actual Total Time"]
        if run_time < 100:
            continue
        plan = plan["plan"][0][0][0]
        plan["database_id"] = 0
        train_plans.append(plan)

    print("generating train encoding...")
    train_plans_meta = []
    for plan in tqdm(train_plans):
        # get plan encoding, plan_meta: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
        plan_mata = get_plan_encoding(
            plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        )
        train_plans_meta.append(plan_mata[:-1])

    train_plans_meta, val_plans_meta = train_test_split(
        train_plans_meta, test_size=0.1, random_state=123
    )
    train_dataset = prepare_dataset(train_plans_meta)
    val_dataset = prepare_dataset(val_plans_meta)
    batch_size = configs["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lora.mark_only_lora_as_trainable(model.model)
    wandb_logger = pl.loggers.WandbLogger(project="DACE-tuning")

    wandb_logger.log_hyperparams(configs)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(ROOT_DIR, "checkpoints"),
        filename="DACE_tuning",
    )
    trainer = PLTrainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    # trainer = PLTrainer(accelerator="cpu", max_epochs=50, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)


# test
def test_job(model, test_workloads):
    statistics_file_path = configs["statistics_path"]
    feature_statistics = load_json(ROOT_DIR + statistics_file_path)
    # add numerical scalers (cite from zero-shot)
    add_numerical_scalers(feature_statistics)

    # op_name to one-hot, using feature_statistics
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    model = model.model
    for test_workload in test_workloads:
        print("test workload: {}".format(test_workload))
        data = load_json(
            os.path.join(ROOT_DIR, "data/workload2/{}_plans.json".format(test_workload))
        )
        plans = []
        for plan in data:
            run_time = plan["plan"][0][0][0]["Plan"]["Actual Total Time"]
            if run_time < 100:
                continue
            plan = plan["plan"][0][0][0]
            plan["database_id"] = 0
            plans.append(plan)

        plans_meta = []
        for plan in tqdm(plans):
            # get plan encoding, plan_meta: (seq_encoding, run_times, attention_mask, loss_mask, database_id)
            plan_mata = get_plan_encoding(
                plan, configs, op_name_to_one_hot, plan_parameters, feature_statistics
            )
            plans_meta.append(plan_mata[:-1])

        test_dataset = prepare_dataset(plans_meta)
        batch_size = configs["batch_size"]
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model.eval()
        preds = []
        labels = []
        for batch in test_dataloader:
            seqs_padded, attn_masks, loss_mask, run_times = batch
            est_run_times = model(seqs_padded, attn_masks)
            est_run_times = est_run_times[:, 0]
            run_times = run_times[:, 0]
            preds.append(est_run_times.detach().cpu().numpy())
            labels.append(run_times.detach().cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        q_errors = q_error_np(preds, labels)
        print_qerrors(q_errors)


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
        default="data/workload2",
        help="plans directory",
    )
    parser.add_argument(
        "--statistics_path",
        type=str,
        default="data/workload1/statistics.json",
        help="statistics file path",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="whether to tune the model (if false, only test the model)",
    )

    args = parser.parse_args()

    # transform args to configs
    configs = vars(args)
    configs["max_runtime"] = 30000

    # set random seed
    set_seed(configs["random_seed"])

    model = DACELora(
        configs["node_length"], configs["hidden_dim"], 1, configs["mlp_activation"]
    )

    getModelSize(model)

    model = PL_DACE(model)

    if not os.path.exists(os.path.join(ROOT_DIR, "checkpoints/DACE_imdb.ckpt")):
        print("please get the pre-trained model first!")
        exit(0)

    model_dict = torch.load(
        os.path.join(ROOT_DIR, "checkpoints/DACE_imdb.ckpt"),
    )
    model.load_state_dict(model_dict["state_dict"])

    if configs["tune"]:
        tune_DACE(configs, model)

    test_workloads = ["synthetic", "scale", "job-light"]
    test_job(model, test_workloads)
