import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import SCENARIOS, INIT_LR, LR_DROP_PERIOD, LR_DROP_FACTOR
from dataset import BeamformingDataset
from models import DNN_NSB, CNN_MVDR


def train_model(scenario_id: int, mode: str = "dnn_nsb", data_dir: str = "data", out_dir: str = "checkpoints"):
    cfg = SCENARIOS[scenario_id]
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {mode.upper()} for {cfg.name} on {device}")

    x_path = os.path.join(data_dir, f"s{scenario_id}_{mode}_X.npy")
    y_path = os.path.join(data_dir, f"s{scenario_id}_{mode}_Y.npy")
    full_ds = BeamformingDataset(x_path, y_path)

    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if mode == "dnn_nsb":
        model = DNN_NSB(cfg.N, cfg.num_layers, cfg.hidden_nodes).to(device)
    else:
        num_conv = 3 if scenario_id == 1 else 8
        model = CNN_MVDR(cfg.N, num_conv_layers=num_conv).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_PERIOD, gamma=LR_DROP_FACTOR)

    train_losses, val_losses = [], []
    best_val = float("inf")
    ckpt_path = os.path.join(out_dir, f"s{scenario_id}_{mode}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_mse = running_loss / train_size

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_running += criterion(preds, yb).item() * xb.size(0)
        val_mse = val_running / val_size

        scheduler.step()
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), ckpt_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}/{cfg.epochs} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | LR: {lr_now:.2e}")

    # plot loss curves
    plt.figure(figsize=(6, 4))
    plt.semilogy(range(1, cfg.epochs + 1), train_losses, "k-", label="Training Loss")
    plt.semilogy(range(1, cfg.epochs + 1), val_losses, "b--", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{mode.upper()} Loss Curve - Scenario {scenario_id}")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"s{scenario_id}_{mode}_loss.png"), dpi=150)
    plt.close()
    print(f"Best model saved to {ckpt_path} (Val MSE: {best_val:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--mode", type=str, default="dnn_nsb", choices=["dnn_nsb", "cnn_mvdr"])
    args = parser.parse_args()
    train_model(args.scenario, mode=args.mode)