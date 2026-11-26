import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import SCENARIOS, K_SAMPLES
from signals import generate_snapshot, extract_autocorr_features
from beamformers import (
    calc_nsb_weights,
    calc_mvdr_weights,
    calc_svd_mvdr_weights,
    target_to_weights,
    compute_sinr_db,
    compute_gain_pattern_db,
)
from dataset import sample_valid_angles
from models import DNN_NSB

def predict_dnn_weights(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    z = extract_autocorr_features(X)
    with torch.no_grad():
        w_hat = model(torch.from_numpy(z).unsqueeze(0).to(device)).cpu().numpy()[0]
    return target_to_weights(w_hat)

PAPER_TEST_CASES = {
    1: {"theta_d": -13.0, "theta_i": [11.0, 30.0],               "snr_db": 5.0, "isr_db": 10.0},
    2: {"theta_d":  40.0, "theta_i": [0.0, 10.0],                "snr_db": 5.0, "isr_db": 20.0},
    3: {"theta_d":  -5.0, "theta_i": [-40.0, 15.0, 25.0, 35.0],  "snr_db": 5.0, "isr_db": 10.0},
    4: {"theta_d": -25.0, "theta_i": [0.0, 20.0],                "snr_db": 5.0, "isr_db": 10.0},
}

def plot_sample_gain_pattern(
    model,
    scenario_id: int,
    device: torch.device,
    ckpt_dir: str = "checkpoints",
    rng: np.random.Generator = None,
):
    cfg = SCENARIOS[scenario_id]
    tc = PAPER_TEST_CASES[scenario_id]
    theta_d, theta_i = tc["theta_d"], tc["theta_i"]
    snr_db, isr_db = tc["snr_db"], tc["isr_db"]

    X, _, _, _ = generate_snapshot(
        theta_d, theta_i, cfg.N, snr_db, isr_db, K=K_SAMPLES, rng=rng
    )
    w_nsb = calc_nsb_weights(theta_d, theta_i, cfg.N)
    w_dnn = predict_dnn_weights(model, X, device)

    angles = np.linspace(-90.0, 90.0, 1801)  # 0.1 deg resolution
    gain_nsb = compute_gain_pattern_db(w_nsb, angles)
    gain_dnn = compute_gain_pattern_db(w_dnn, angles)

    # print null depths and local minimum angles near each jammer
    print(f"\n--- Scenario {scenario_id} Gain Pattern Check (theta_d={theta_d}°, ISR={isr_db} dB) ---")
    for th in theta_i:
        idx = int(np.argmin(np.abs(angles - th)))
        # search +/- 2 deg window for DNN's actual null peak
        win = np.where(np.abs(angles - th) <= 2.0)[0]
        dnn_min_idx = win[np.argmin(gain_dnn[win])]
        print(
            f"Jammer at {th:5.1f}° | "
            f"Calc NSB: {gain_nsb[idx]:6.1f} dB | "
            f"DNN-NSB at {th:5.1f}°: {gain_dnn[idx]:6.1f} dB "
            f"(min {gain_dnn[dnn_min_idx]:6.1f} dB @ {angles[dnn_min_idx]:5.1f}°)"
        )

    # plot matching Figs. 7, 10, 14
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(angles, gain_nsb, "k-", linewidth=1.2, label="Calculated NSB")
    plt.plot(angles, gain_dnn, "b--", linewidth=1.3, label="DNN-NSB")

    plt.axvline(theta_d, color="gray", linestyle="--", linewidth=0.9)
    plt.text(theta_d, 3.0, f"{theta_d:g}°", ha="center", va="bottom", fontsize=9)
    for th in theta_i:
        plt.axvline(th, color="gray", linestyle=":", linewidth=0.9)
        plt.text(th, 3.0, f"{th:g}°", ha="center", va="bottom", fontsize=9)

    plt.xlim(-90, 90)
    plt.ylim(-80, 10)
    plt.xticks(np.arange(-90, 91, 30))
    plt.xlabel("Angle (deg)")
    plt.ylabel("Gain (dB)")
    plt.title(f"Scenario {scenario_id}: Gain Pattern (SNR={snr_db:g} dB, ISR={isr_db:g} dB)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")
    plt.tight_layout()

    out_path = os.path.join(ckpt_dir, f"s{scenario_id}_gain_pattern.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved gain pattern to {out_path}")
    plt.show()

def evaluate_scenario(scenario_id: int, ckpt_dir: str = "checkpoints", num_trials: int = 500):
    cfg = SCENARIOS[scenario_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(123)

    model = DNN_NSB(cfg.N, cfg.num_layers, cfg.hidden_nodes).to(device)
    ckpt_path = os.path.join(ckpt_dir, f"s{scenario_id}_dnn_nsb.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 1. SINR vs SNR sweep
    snr_grid = np.arange(0, 11, 2, dtype=np.float64)
    sinr_snr = {k: [] for k in ["nsb", "dnn", "mvdr", "svd"]}
    for snr_db in snr_grid:
        vals = {k: [] for k in sinr_snr}
        for _ in range(num_trials):
            theta_d, theta_i = sample_valid_angles(cfg, rng)
            X, _, X_i, X_n = generate_snapshot(theta_d, theta_i, cfg.N, snr_db, 10.0, K=K_SAMPLES, rng=rng)
            w_nsb = calc_nsb_weights(theta_d, theta_i, cfg.N)
            w_dnn = predict_dnn_weights(model, X, device)
            w_mvdr = calc_mvdr_weights(X_i + X_n, theta_d, cfg.N)
            w_svd = calc_svd_mvdr_weights(X_i + X_n, theta_d, cfg.N, cfg.Q)

            vals["nsb"].append(compute_sinr_db(w_nsb, theta_d, theta_i, snr_db, 10.0))
            vals["dnn"].append(compute_sinr_db(w_dnn, theta_d, theta_i, snr_db, 10.0))
            vals["mvdr"].append(compute_sinr_db(w_mvdr, theta_d, theta_i, snr_db, 10.0))
            vals["svd"].append(compute_sinr_db(w_svd, theta_d, theta_i, snr_db, 10.0))
        for k in sinr_snr:
            sinr_snr[k].append(np.mean(vals[k]))

    # 2. SINR vs ISR sweep
    isr_grid = np.array([2.5, 5.0, 10.0, 15.0, 20.0])
    sinr_isr = {k: [] for k in ["nsb", "dnn", "mvdr", "svd"]}
    for isr_db in isr_grid:
        vals = {k: [] for k in sinr_isr}
        for _ in range(num_trials):
            theta_d, theta_i = sample_valid_angles(cfg, rng)
            X, _, X_i, X_n = generate_snapshot(theta_d, theta_i, cfg.N, 5.0, isr_db, K=K_SAMPLES, rng=rng)
            w_nsb = calc_nsb_weights(theta_d, theta_i, cfg.N)
            w_dnn = predict_dnn_weights(model, X, device)
            w_mvdr = calc_mvdr_weights(X_i + X_n, theta_d, cfg.N)
            w_svd = calc_svd_mvdr_weights(X_i + X_n, theta_d, cfg.N, cfg.Q)

            vals["nsb"].append(compute_sinr_db(w_nsb, theta_d, theta_i, 5.0, isr_db))
            vals["dnn"].append(compute_sinr_db(w_dnn, theta_d, theta_i, 5.0, isr_db))
            vals["mvdr"].append(compute_sinr_db(w_mvdr, theta_d, theta_i, 5.0, isr_db))
            vals["svd"].append(compute_sinr_db(w_svd, theta_d, theta_i, 5.0, isr_db))
        for k in sinr_isr:
            sinr_isr[k].append(np.mean(vals[k]))

    # plot SINR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(snr_grid, sinr_snr["nsb"], "k-", label="Calculated NSB")
    axes[0].plot(snr_grid, sinr_snr["dnn"], "b--", label="DNN-NSB")
    axes[0].plot(snr_grid, sinr_snr["mvdr"], "m:", label="MVDR")
    axes[0].plot(snr_grid, sinr_snr["svd"], "c-.", label="SVD-MVDR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("SINR (dB)")
    axes[0].set_title(f"Scenario {scenario_id}: SINR vs SNR (ISR=10 dB)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(isr_grid, sinr_isr["nsb"], "k-s", label="Calculated NSB")
    axes[1].plot(isr_grid, sinr_isr["dnn"], "b--o", label="DNN-NSB")
    axes[1].plot(isr_grid, sinr_isr["mvdr"], "m:^", label="MVDR")
    axes[1].plot(isr_grid, sinr_isr["svd"], "c-.v", label="SVD-MVDR")
    axes[1].set_xlabel("ISR (dB)")
    axes[1].set_ylabel("SINR (dB)")
    axes[1].set_title(f"Scenario {scenario_id}: SINR vs ISR (SNR=5 dB)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, f"s{scenario_id}_sinr_curves.png"), dpi=150)
    plt.show()
    plot_sample_gain_pattern(model, scenario_id, device, ckpt_dir=ckpt_dir, rng=rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()
    evaluate_scenario(args.scenario, num_trials=args.trials)