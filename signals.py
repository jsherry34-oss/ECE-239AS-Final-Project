import numpy as np
from config import D_LAMBDA, FS, K_SAMPLES


def steering_vector(theta_deg: float, N: int, d_lambda: float = D_LAMBDA) -> np.ndarray:
    """
    ULA steering vector a_s(theta) of shape (N, 1)
    """
    theta_rad = np.deg2rad(theta_deg)
    n = np.arange(N, dtype=np.float64)[:, None]
    beta = 2.0 * np.pi * d_lambda * n * np.sin(theta_rad)
    return np.exp(-1j * beta)


def generate_lfm_pulse(
    K: int = K_SAMPLES,
    fs: float = FS,
    pulse_start: int = 40,
    pulse_len: int = 40,
    bandwidth: float = 0.4e6,
) -> np.ndarray:
    """
    generate a unit-power complex baseband LFM radar pulse
    """
    s_d = np.zeros(K, dtype=np.complex128)
    T_p = pulse_len / fs
    t = np.arange(pulse_len) / fs - (T_p / 2.0)
    chirp_rate = bandwidth / T_p
    pulse = np.exp(1j * np.pi * chirp_rate * (t ** 2))
    s_d[pulse_start : pulse_start + pulse_len] = pulse
    s_d = s_d / np.sqrt(np.mean(np.abs(s_d) ** 2))
    return s_d[None, :]


def generate_wideband_jammers(Q: int, K: int = K_SAMPLES, rng: np.random.Generator = None) -> np.ndarray:
    """
    generate Q independent unit-variance complex Gaussian wideband jamming signals
    """
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal((Q, K))
    imag = rng.standard_normal((Q, K))
    return (real + 1j * imag) / np.sqrt(2.0)


def generate_awgn(N: int, K: int = K_SAMPLES, rng: np.random.Generator = None) -> np.ndarray:
    """
    generates i.i.d. unit-variance complex AWGN across N array elements and K samples
    """
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal((N, K))
    imag = rng.standard_normal((N, K))
    return (real + 1j * imag) / np.sqrt(2.0)


def generate_snapshot(
    theta_d: float,
    theta_i_list: list,
    N: int,
    snr_db: float,
    isr_db: float,
    K: int = K_SAMPLES,
    rng: np.random.Generator = None,
):
    """
    generate a full received snapshot and return individual components
    """
    if rng is None:
        rng = np.random.default_rng()

    Q = len(theta_i_list)
    snr_lin = 10.0 ** (snr_db / 10.0)
    isr_lin = 10.0 ** (isr_db / 10.0)

    # active pulse power
    s_d = generate_lfm_pulse(K=K)
    a_d = steering_vector(theta_d, N)
    X_d = a_d @ s_d  # (N, K)

    # interference power
    s_i = generate_wideband_jammers(Q, K=K, rng=rng) * np.sqrt(isr_lin)
    A_i = np.hstack([steering_vector(th, N) for th in theta_i_list])  # (N, Q)
    X_i = A_i @ s_i  # (N, K)

    # noise variance
    sigma_n = np.sqrt(1.0 / snr_lin)
    X_n = generate_awgn(N, K=K, rng=rng) * sigma_n  # (N, K)

    X = X_d + X_i + X_n
    return X, X_d, X_i, X_n


def extract_autocorr_features(X: np.ndarray) -> np.ndarray:
    """
    compute sample autocorrelation matrix, extracts upper triangular
    elements, splits real/imag, and normalizes
    """
    N, K = X.shape
    R = (X @ X.conj().T) / K
    triu_idx = np.triu_indices(N)
    r_tilde = R[triu_idx]
    r_hat = np.concatenate([np.real(r_tilde), np.imag(r_tilde)])
    norm = np.linalg.norm(r_hat)
    z = r_hat / (norm + 1e-12)
    return z.astype(np.float32)