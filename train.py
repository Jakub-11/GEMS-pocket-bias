"""Train one GATE18d model on one split.

    python train.py --model ood_1nvq --fold 2
    python train.py --model cleansplit --fold 0 --devices 8

Writes <RUNS>/<model>_f<fold>/:
    checkpoints/best.ckpt     lowest val_rmse  — this is what the ensemble uses
    checkpoints/last.ckpt
    metrics.csv               per-epoch train/val curves

Config is fixed to the published setup: SGD(m=0.9) lr 1e-3, wd 1e-3, RMSE,
dropout 0.5, batch 32 per device (x8 GPUs = 256 effective), max 2000 epochs,
early stopping on val_rmse (patience 100, tolerance 1%), seed 0.
"""
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gems import paths
from gems.data import SplitData
from gems.model import GATE18d, EarlyStopOnRMSE


def main():
    ap = argparse.ArgumentParser(description="Train one GATE18d model on one split.")
    ap.add_argument("--model", required=True, choices=paths.ALL_MODELS)
    ap.add_argument("--fold", type=int, required=True, choices=range(paths.N_FOLDS))
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32, help="per device")
    ap.add_argument("--max_epochs", type=int, default=2000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    out = paths.run_dir(args.model, args.fold)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    data = SplitData(args.model, args.fold, args.batch_size, args.num_workers)
    data.setup()
    in_channels, edge_dim = data.feature_dims()
    print(f"[train] {args.model} f{args.fold}: in_channels={in_channels} edge_dim={edge_dim}", flush=True)

    model = GATE18d(in_channels=in_channels, edge_dim=edge_dim)

    best = ModelCheckpoint(dirpath=out / "checkpoints", filename="best",
                           monitor="val_rmse", mode="min", save_top_k=1, save_last=True)
    gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if gpu else "cpu",
        devices=args.devices if gpu else 1,
        strategy="ddp" if (gpu and args.devices > 1) else "auto",
        sync_batchnorm=(gpu and args.devices > 1),
        max_epochs=args.max_epochs,
        callbacks=[best, EarlyStopOnRMSE(patience=100, tolerance=0.01)],
        logger=CSVLogger(save_dir=str(out), name="", version=""),
        default_root_dir=str(out),
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data)

    if trainer.is_global_zero:
        # marker so downstream stages can tell a finished run from a running one
        (out / "DONE").write_text(f"{best.best_model_path}\n{best.best_model_score:.4f}\n")
        print(f"[train] done: best val_rmse ckpt -> {best.best_model_path} "
              f"(val_rmse={best.best_model_score:.4f})", flush=True)


if __name__ == "__main__":
    main()
