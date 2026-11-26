import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import SCENARIOS, K_SAMPLES
from signals import generate_snapshot, extract_autocorr_features
from beamformers import calc_nsb_weights, calc_mvdr_weights, weights_to_target


def sample_valid_angles(cfg, rng: np.random.Generator):
    """
    uniformly sample discrete DOAs for desired signal and Q interferers
    ensuring no two angles collide
    """
    d_grid = np.arange(cfg.theta_d_range[0], cfg.theta_d_range[1] + 1, cfg.theta_d_step, dtype=np.float64)
    i_grid = np.arange(cfg.theta_i_range[0], cfg.theta_i_range[1] + 1, cfg.theta_i_step, dtype=np.float64)
    num_var_i = cfg.Q - len(cfg.fixed_interferers)

    while True:
        theta_d = float(rng.choice(d_grid))
        var_i = [float(x) for x in rng.choice(i_grid, size=num_var_i, replace=False)]
        theta_i = var_i + list(cfg.fixed_interferers)
        all_angles = [theta_d] + theta_i
        # Require distinct angles
        if len(set(all_angles)) == len(all_angles):
            return theta_d, theta_i


def generate_dataset(
    scenario_id: int,
    num_samples: int = None,
    save_dir: str = "data",
    mode: str = "dnn_nsb",
    seed: int = 42,
):
    """
    generate and save synthetic training/validation data for a given scenario
    """
    cfg = SCENARIOS[scenario_id]
    total = num_samples if num_samples is not None else cfg.num_samples
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    N = cfg.N
    if mode == "dnn_nsb":
        X_feat = np.zeros((total, N * (N + 1)), dtype=np.float32)
    elif mode == "cnn_mvdr":
        X_feat = np.zeros((total, 2, N, N), dtype=np.float32)
    else:
        raise ValueError("mode must be 'dnn_nsb' or 'cnn_mvdr'")

    Y_targ = np.zeros((total, 2 * N), dtype=np.float32)

    for idx in tqdm(range(total), desc=f"Generating S{scenario_id} ({mode})"):
        theta_d, theta_i = sample_valid_angles(cfg, rng)
        isr_db = float(rng.choice(cfg.isr_values_db))

        X, _, X_i, X_n = generate_snapshot(
            theta_d, theta_i, N, cfg.snr_train_db, isr_db, K=K_SAMPLES, rng=rng
        )

        if mode == "dnn_nsb":
            X_feat[idx] = extract_autocorr_features(X)
            w = calc_nsb_weights(theta_d, theta_i, N)
        else:
            R = (X @ X.conj().T) / K_SAMPLES
            R_mat = np.stack([np.real(R), np.imag(R)], axis=0)
            norm = np.linalg.norm(R_mat)
            X_feat[idx] = (R_mat / (norm + 1e-12)).astype(np.float32)
            w = calc_mvdr_weights(X_i + X_n, theta_d, N)

        Y_targ[idx] = weights_to_target(w)

    x_path = os.path.join(save_dir, f"s{scenario_id}_{mode}_X.npy")
    y_path = os.path.join(save_dir, f"s{scenario_id}_{mode}_Y.npy")
    np.save(x_path, X_feat)
    np.save(y_path, Y_targ)
    print(f"Saved {total} samples to {x_path} and {y_path}")


class BeamformingDataset(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.X[idx], copy=True))
        y = torch.from_numpy(np.array(self.Y[idx], copy=True))
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--mode", type=str, default="dnn_nsb", choices=["dnn_nsb", "cnn_mvdr"])
    parser.add_argument("--num_samples", type=int, default=None, help="Override default sample count for quick tests")
    args = parser.parse_args()
    generate_dataset(args.scenario, num_samples=args.num_samples, mode=args.mode)