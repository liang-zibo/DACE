import torch
import torch.nn as nn
import lightning.pytorch as pl
import loralib as lora
from plan_utils import q_error


# create DACE model with lora
class DACELora(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.3,
        transformer_dropout=0.2,
    ):
        super(DACELora, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                dim_feedforward=hidden_dim,
                nhead=1,
                batch_first=True,
                activation=transformer_activation,
                dropout=transformer_dropout,
            ),
            num_layers=1,
        )
        self.node_length = input_dim
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        self.mlp_hidden_dims = [128, 64, 1]
        self.mlp = nn.Sequential(
            *[
                lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=16),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=8),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                lora.Linear(self.mlp_hidden_dims[1], output_dim, r=4),
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attn_mask=None):
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 18 bits
        x = x.view(x.shape[0], -1, self.node_length)
        out = self.tranformer_encoder(x, mask=attn_mask)
        out = self.mlp(out)
        out = self.sigmoid(out).squeeze(dim=2)
        return out


class EncoderFormer(nn.Module):
    """
    As a pre-trained encoder.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.3,
        transformer_dropout=0.2,
    ):
        super(EncoderFormer, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                dim_feedforward=hidden_dim,
                nhead=1,
                batch_first=True,
                activation=transformer_activation,
                dropout=transformer_dropout,
            ),
            num_layers=1,
        )
        self.node_length = input_dim
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        self.mlp_hidden_dims = [128, 64, 1]
        self.mlp = nn.Sequential(
            *[
                lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=32),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=16),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attn_mask=None):
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        x = x.view(x.shape[0], -1, self.node_length)
        out = self.tranformer_encoder(x, mask=attn_mask)
        out = self.mlp(out[:, 0, :])
        return out


# create pytorch_lightning model, support DACE, alter the loss function
class PL_DACE(pl.LightningModule):
    def __init__(self, model):
        super(PL_DACE, self).__init__()
        self.model = model

    def forward(self, x, attn_mask=None):
        return self.model(x, attn_mask)

    def DACE_loss(self, est_run_times, run_times, loss_mask):
        # est_run_times: (batch, seq_len)
        # run_times: (batch, seq_len)
        # seqs_length: (batch,)
        # return: loss (batch,)
        # don't calculate the loss of padding nodes, set them to 0
        loss = torch.max(est_run_times / run_times, run_times / est_run_times)
        loss = loss * loss_mask
        loss = torch.log(torch.where(loss > 1, loss, 1))
        loss = torch.sum(loss, dim=1)
        return loss

    def training_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        loss = self.DACE_loss(est_run_times, run_times, loss_mask)
        loss = torch.mean(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        est_run_times = est_run_times[:, 0]
        run_times = run_times[:, 0]
        # calculate q-error
        qerror = q_error(est_run_times, run_times)
        loss = torch.mean(qerror)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        loss = self.DACE_loss(est_run_times, run_times, loss_mask)
        loss = torch.mean(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# create pytorch_lightning trainer and overwrite the test function
class PLTrainer(pl.Trainer):
    # succeed the init function
    def __init__(self, *args, **kwargs):
        super(PLTrainer, self).__init__(*args, **kwargs)

    def test(self, model, dataloaders=None, ckpt_path=None):
        if dataloaders is None:
            if self.test_dataloaders is None:
                raise ValueError(
                    "Trainer that returned None for test_dataloaders or passed None to test"
                )
            dataloaders = self.test_dataloaders

        model.eval()

        # get q-error of all test data
        qerrors = []
        for batch, attn_masks, loss_mask, batch_times in dataloaders:
            est_times = model(batch, attn_masks)
            # use for DACE
            est_times = est_times[:, 0]
            batch_times = batch_times[:, 0]
            # calculate q-error
            qerror = q_error(est_times, batch_times)
            qerrors.append(qerror)
        qerrors = torch.cat(qerrors, dim=0)
        # save test loss, median, 90th, 95th, 99th, max and mean in a dict
        test_metrics = {}
        test_metrics["50th test loss"] = torch.quantile(qerrors, 0.5).item()
        test_metrics["90th test loss"] = torch.quantile(qerrors, 0.9).item()
        test_metrics["95th test loss"] = torch.quantile(qerrors, 0.95).item()
        test_metrics["mean test loss"] = torch.mean(qerrors).item()
        # report test loss, median, 90th, 95th, 99th, max and mean in logger
        for k, v in test_metrics.items():
            self.logger.log_metrics({k: v})
        return test_metrics
