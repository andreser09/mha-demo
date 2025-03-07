import torch
import pytorch_lightning as pl
from CMAPSSDataset import CMAPSSDataset
from attn_lstm import MultiHeadAttentionLSTM
from torch.nn import functional as F
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.functional import mean_squared_error
from metrics import score  # Assuming this is your custom metric function


class Module(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for logging/checkpointing
        self.net = MultiHeadAttentionLSTM(**kwargs)
        self.lr = lr
        self.test_outputs = []  # Initialize list to store test step outputs

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_rmse", torch.sqrt(loss), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y, id = batch
        y_hat = self.net(x)
        # Append outputs to the list
        self.test_outputs.append({"id": id, "y_hat": y_hat, "y": y})

    def on_test_epoch_start(self):
        # Clear the list at the start of each test epoch
        self.test_outputs = []

    def on_test_epoch_end(self):
        # Aggregate all test step outputs
        ids = torch.cat([x["id"] for x in self.test_outputs])
        y_hats = torch.cat([x["y_hat"] for x in self.test_outputs])
        ys = torch.cat([x["y"] for x in self.test_outputs])
        
        # Compute RMSE on GPU
        rmse = torch.sqrt(mean_squared_error(y_hats, ys))
        self.log("test_rmse", rmse, on_epoch=True, logger=True)

        # Move tensors to CPU for the score function (assuming it uses NumPy)
        y_hats_cpu = y_hats.cpu().numpy()
        ys_cpu = ys.cpu().numpy()
        s = score(y_hats_cpu, ys_cpu)  # Call score with CPU-based NumPy arrays
        
        self.log("test_score", s, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Lightning CMAPSS Example")
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--feature-num", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=100, help="RNN hidden dims")
    parser.add_argument("--cell", type=str, default="lstm", help="lstm, gru or rnn")
    parser.add_argument("--fc-layer-dim", type=int, default=100)
    parser.add_argument("--rnn-num-layers", type=int, default=3)
    parser.add_argument("--fc-activation", type=str, default="relu", help="relu, tanh or gelu")
    parser.add_argument("--attention-order", action="append", help='value must be "feature"')
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--feature-head-num", type=int, default=4)
    parser.add_argument("--fc-dropout", type=float, default=0.5)
    parser.add_argument("--save-attention-weights", action="store_true", default=False)
    parser.add_argument("--dataset-root", type=str, required=True, help="The dir of CMAPSS dataset")
    parser.add_argument("--sub-dataset", type=str, required=True, help="FD001/2/3/4")
    parser.add_argument("--norm-type", type=str, default='z-score', help="z-score, -1-1 or 0-1")
    parser.add_argument("--max-rul", type=int, default=125, help="piece-wise RUL")
    parser.add_argument("--cluster-operations", action="store_true", default=False)
    parser.add_argument("--norm-by-operations", action="store_true", default=False)
    parser.add_argument("--use-max-rul-on-test", action="store_true", default=True)
    parser.add_argument("--validation-rate", type=float, default=0, help="validation set ratio of train set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=50, help="Early Stop Patience")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    args = parser.parse_args()

    model_kwargs = {
        "sequence_len": args.sequence_len,
        "feature_num": args.feature_num,
        "hidden_dim": args.hidden_dim,
        "cell": args.cell,
        "fc_layer_dim": args.fc_layer_dim,
        "rnn_num_layers": args.rnn_num_layers,
        "output_dim": 1,
        "fc_activation": args.fc_activation,
        "attention_order": args.attention_order or [],
        "bidirectional": args.bidirectional,
        "feature_head_num": args.feature_head_num,
        "fc_dropout": args.fc_dropout,
    }

    train_loader, test_loader, valid_loader = CMAPSSDataset.get_data_loaders(
        dataset_root=args.dataset_root,
        sequence_len=args.sequence_len,
        sub_dataset=args.sub_dataset,
        norm_type=args.norm_type,
        max_rul=args.max_rul,
        cluster_operations=args.cluster_operations,
        norm_by_operations=args.norm_by_operations,
        use_max_rul_on_test=args.use_max_rul_on_test,
        validation_rate=args.validation_rate,
        return_id=True,
        use_only_final_on_test=not args.save_attention_weights,
        loader_kwargs={"batch_size": args.batch_size},
    )

    model = Module(lr=args.lr, **model_kwargs)

    early_stop_callback = EarlyStopping(
        monitor="val_rmse",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_rmse",
        filename="checkpoint-{epoch:02d}-{val_rmse:.4f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        default_root_dir="./checkpoints",
        accelerator="gpu" if torch.cuda.is_available() and not args.no_cuda else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_dataloaders=valid_loader or test_loader)
    trainer.test(dataloaders=test_loader)