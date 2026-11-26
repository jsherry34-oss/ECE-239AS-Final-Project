import numpy as np
from signals import steering_vector


def calc_nsb_weights(theta_d: float, theta_i_list: list, N: int) -> np.ndarray:
    """
    analytical nsb weights
    """
    cols = [steering_vector(theta_d, N)] + [steering_vector(th, N) for th in theta_i_list]
    A = np.hstack(cols)  # (N, Q+1)
    g = np.zeros((A.shape[1], 1), dtype=np.complex128)
    g[0, 0] = 1.0
    AHA = A.conj().T @ A
    w = A @ np.linalg.solve(AHA, g)
    return w


def calc_mvdr_weights(X_interf_noise: np.ndarray, theta_d: float, N: int, diag_load: float = 1e-6) -> np.ndarray:
    """
    analytical mvdr weights
    """
    K = X_interf_noise.shape[1]
    C_s = (X_interf_noise @ X_interf_noise.conj().T) / K
    C_s = C_s + diag_load * np.eye(N, dtype=np.complex128)
    a_d = steering_vector(theta_d, N)
    C_inv_a = np.linalg.solve(C_s, a_d)
    w = C_inv_a / (a_d.conj().T @ C_inv_a)
    return w


def calc_svd_mvdr_weights(X_interf_noise: np.ndarray, theta_d: float, N: int, Q: int) -> np.ndarray:
    """
    analytical svd mvdr
    """
    K = X_interf_noise.shape[1]
    C_s = (X_interf_noise @ X_interf_noise.conj().T) / K
    U, s, Vh = np.linalg.svd(C_s)
    s_clean = s.copy()
    if Q < N:
        noise_floor = np.mean(s[Q:])
        s_clean[Q:] = noise_floor
    C_clean = (U * s_clean) @ Vh
    a_d = steering_vector(theta_d, N)
    C_inv_a = np.linalg.solve(C_clean, a_d)
    w = C_inv_a / (a_d.conj().T @ C_inv_a)
    return w


def weights_to_target(w: np.ndarray) -> np.ndarray:
    """
    convert complex weight vector (N, 1) to normalized real/imag vector (2N,)
    """
    w_flat = w.reshape(-1)
    w_hat = np.concatenate([np.real(w_flat), np.imag(w_flat)])
    w_hat = w_hat / (np.linalg.norm(w_hat) + 1e-12)
    return w_hat.astype(np.float32)


def target_to_weights(w_hat: np.ndarray) -> np.ndarray:
    """
    converts real/imag vector (2N,) back to complex weight vector (N, 1)
    """
    N = w_hat.shape[0] // 2
    return (w_hat[:N] + 1j * w_hat[N:])[:, None]


def compute_sinr_db(
    w: np.ndarray,
    theta_d: float,
    theta_i_list: list,
    snr_db: float,
    isr_db: float,
) -> float:
    """
    compute output SINR (dB) for weight vector w given array geometry and SNR/ISR.
    """
    N = w.shape[0]
    snr_lin = 10.0 ** (snr_db / 10.0)
    isr_lin = 10.0 ** (isr_db / 10.0)
    sigma_n2 = 1.0 / snr_lin

    a_d = steering_vector(theta_d, N)
    p_sig = np.abs((w.conj().T @ a_d).item()) ** 2

    p_int = 0.0
    for th in theta_i_list:
        a_i = steering_vector(th, N)
        p_int += isr_lin * (np.abs((w.conj().T @ a_i).item()) ** 2)

    p_noise = sigma_n2 * (np.linalg.norm(w) ** 2)
    return 10.0 * np.log10(p_sig / (p_int + p_noise + 1e-15))


def compute_gain_pattern_db(w: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    compute normalized array gain pattern
    """
    N = w.shape[0]
    A_grid = np.hstack([steering_vector(th, N) for th in angles_deg])  # (N, M)
    resp = np.abs((w.conj().T @ A_grid).reshape(-1))
    resp_db = 20.0 * np.log10(resp / (np.max(resp) + 1e-12) + 1e-12)
    return resp_db