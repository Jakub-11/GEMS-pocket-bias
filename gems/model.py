"""GATE18d — the GEMS18d binding-affinity GNN as a PyTorch-Lightning module.

Architecture (unchanged from GEMS18d):

    x  ──► FeatureTransformMLP ──┐
    edge_attr ───────────────────┼──► MetaLayer1 ──► BN ──► MetaLayer2 ──► dropout ──► head ──► pK
    lig_emb (ChemBERTa, 384) ────┘        (global feature u initialised from lig_emb)

Each MetaLayer is {EdgeModel MLP, NodeModel GATv2Conv(4 heads), GlobalModel MLP}.
Labels are scaled to [0,1]; predictions are multiplied by `scale_max`(=16) to get pK.

Training config is fixed to the one used for every published run: SGD(momentum
0.9), lr 1e-3, weight decay 1e-3, RMSE loss, dropout 0.5, no gradient clipping.

`__init__` accepts (and ignores) extra keyword arguments so that checkpoints
written by the earlier, more configurable version of this class still load.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric.nn as geom_nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_add_pool

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:                                    # scipy lives in the container
    pearsonr = spearmanr = None


# ------------------------------------------------------------------ building blocks
class FeatureTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        return self.dropout_layer(self.mlp(node_features))


class EdgeModel(nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_f + n_edge_f, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        return self.edge_mlp(self.dropout_layer(torch.cat([src, dest, edge_attr], 1)))


class NodeModel(nn.Module):
    def __init__(self, n_node_f, n_edge_f, out_dim, dropout):
        super().__init__()
        self.heads = 4
        self.conv = GATv2Conv(n_node_f, int(out_dim / self.heads), edge_dim=n_edge_f,
                              heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return F.relu(self.conv(x, edge_index, edge_attr))


class GlobalModel(nn.Module):
    def __init__(self, n_node_f, glob_f_in, glob_f_hidden, glob_f_out, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + glob_f_in, glob_f_hidden), nn.ReLU(), nn.Linear(glob_f_hidden, glob_f_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        return self.global_mlp(self.dropout_layer(out))


def _metalayer(node_f, node_f_out, edge_f, edge_f_out, glob_f, glob_f_out, dropout):
    return geom_nn.MetaLayer(
        edge_model=EdgeModel(node_f, edge_f, 64, edge_f_out, dropout),
        node_model=NodeModel(node_f, edge_f_out, node_f_out, dropout),
        global_model=GlobalModel(node_f_out, glob_f, glob_f_out, glob_f_out, dropout))


class GemsHead(nn.Module):
    """GEMS18d head: Linear -> ReLU -> Linear. Dropout is applied by the parent."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, targets, output):
        return torch.sqrt(self.mse(output, targets))


# ------------------------------------------------------------------------- model
class GATE18d(pl.LightningModule):
    HIDDEN_NODE = 64

    def __init__(self, in_channels, edge_dim, lig_emb_dim=384, glob_dim=384,
                 dropout=0.5, conv_dropout=0.0, learning_rate=1e-3, weight_decay=1e-3,
                 scale_max=16.0, label_key="y", lig_emb_key="lig_emb", **_ignored):
        super().__init__()
        self.save_hyperparameters(ignore=list(_ignored))
        h = self.HIDDEN_NODE
        self.lig_emb_dim, self.glob_dim = lig_emb_dim, glob_dim
        self.label_key, self.lig_emb_key = label_key, lig_emb_key
        self.scale_max = scale_max
        self.learning_rate, self.weight_decay = learning_rate, weight_decay

        self.NodeTransform = FeatureTransformMLP(in_channels, 256, h, dropout=dropout)
        self.layer1 = _metalayer(h, h, edge_dim, 64, lig_emb_dim, glob_dim, conv_dropout)
        self.node_bn1 = BatchNorm1d(h)
        self.edge_bn1 = BatchNorm1d(64)
        self.u_bn1 = BatchNorm1d(glob_dim)
        self.layer2 = _metalayer(h, h, 64, 64, glob_dim, glob_dim, conv_dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.reg_head = GemsHead(glob_dim, h)

        self.loss_fn = RMSELoss()
        self._train_out, self._val_out = [], []

    # ---------------------------------------------------------------- forward
    def forward(self, batch):
        x = self.NodeTransform(batch.x)
        u = getattr(batch, self.lig_emb_key)
        x, edge_attr, u = self.layer1(x, batch.edge_index, batch.edge_attr, u=u, batch=batch.batch)
        x, edge_attr, u = self.node_bn1(x), self.edge_bn1(edge_attr), self.u_bn1(u)
        _, _, u = self.layer2(x, batch.edge_index, edge_attr, u, batch=batch.batch)
        u = self.dropout_layer(u)
        return self.reg_head(u)

    def predict_pk(self, batch):
        """Unscaled pK prediction, shape [n_graphs]."""
        return self(batch).view(-1) * self.scale_max

    # ------------------------------------------------------------------ steps
    def _step(self, batch):
        targets = getattr(batch, self.label_key).view(-1, 1)
        preds = self(batch).view(-1, 1)
        return self.loss_fn(targets, preds), preds.detach(), targets.detach()

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self._train_out.append({"loss": loss.detach(), "preds": preds, "targets": targets})
        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch)
        self._val_out.append({"loss": loss.detach(), "preds": preds, "targets": targets})
        return loss

    def on_train_epoch_end(self):
        self._reduce(self._train_out, "train")

    def on_validation_epoch_end(self):
        self._reduce(self._val_out, "val")

    def _reduce(self, outputs, prefix):
        if not outputs:
            return
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean()
        labels = torch.cat([o["targets"] for o in outputs]).view(-1)
        preds = torch.cat([o["preds"] for o in outputs]).view(-1)
        labels = self.all_gather(labels).reshape(-1).cpu().numpy()
        preds = self.all_gather(preds).reshape(-1).cpu().numpy()
        self.log(f"{prefix}_loss", avg_loss, sync_dist=True, prog_bar=True)

        m = self._metrics(labels, preds) if self.trainer.is_global_zero else {"pearson": 0.0, "rmse": 0.0}
        # val_rmse drives early stopping + checkpointing, so it must agree on every rank
        rmse = torch.tensor(m["rmse"], device=self.device, dtype=torch.float32)
        if self.trainer.world_size > 1:
            torch.distributed.broadcast(rmse, src=0)
        self.log(f"{prefix}_rmse", rmse, sync_dist=False, prog_bar=(prefix == "val"))
        if self.trainer.is_global_zero:
            self.log(f"{prefix}_pearson", m["pearson"], rank_zero_only=True)
        outputs.clear()

    def _metrics(self, labels, preds):
        if len(labels) < 2 or np.std(preds) < 1e-9:
            return {"pearson": 0.0, "rmse": 0.0}
        r = pearsonr(labels, preds)[0] if pearsonr else float(np.corrcoef(labels, preds)[0, 1])
        return {"pearson": float(r),
                "rmse": float(np.sqrt(np.mean(((preds - labels) * self.scale_max) ** 2)))}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate,
                               momentum=0.9, weight_decay=self.weight_decay)


class EarlyStopOnRMSE(pl.Callback):
    """GEMS's early stopper: count epochs where val_rmse stays above
    min_rmse * (1 + tolerance); stop after `patience` of them."""

    def __init__(self, patience=100, tolerance=0.01):
        super().__init__()
        self.patience, self.tolerance = patience, tolerance
        self.counter, self.best = 0, float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or "val_rmse" not in trainer.callback_metrics:
            return
        val = float(trainer.callback_metrics["val_rmse"])
        if val < self.best:
            self.best, self.counter = val, 0
        elif val > self.best * (1 + self.tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                if trainer.is_global_zero:
                    print(f"[early-stop] val_rmse {val:.4f} above {self.best * (1 + self.tolerance):.4f} "
                          f"for {self.patience} epochs", flush=True)
                trainer.should_stop = True
