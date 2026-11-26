from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ScenarioConfig:
    name: str
    N: int                          # number of ULA elements
    Q: int                          # number of interference sources
    theta_d_range: Tuple[int, int]  # (min, max) degrees inclusive
    theta_d_step: int               # step size in degrees
    theta_i_range: Tuple[int, int]  # (min, max) degrees inclusive
    theta_i_step: int               # step size in degrees
    fixed_interferers: List[float]  # any fixed interference DOAs
    isr_values_db: List[float]      # training ISR values in dB
    snr_train_db: float = 5.0       # training SNR in dB
    num_layers: int = 3
    hidden_nodes: int = 256
    num_samples: int = 1_000_000
    batch_size: int = 1024
    epochs: int = 50


# global radar signal params
FS = 4e6            # 4 MHz sampling frequency
RX_INTERVAL = 50e-6 # 50 us reception interval
K_SAMPLES = 200     # FS * RX_INTERVAL = 200 samples
D_LAMBDA = 0.5      # Element spacing d = lambda / 2

# training optimizer defaults
INIT_LR = 1e-3
LR_DROP_PERIOD = 5
LR_DROP_FACTOR = 0.70
DROPOUT_RATE = 0.50

SCENARIOS = {
    1: ScenarioConfig(
        name="Scenario 1 (N=8, Q=2, 1 fixed)",
        N=8, Q=2,
        theta_d_range=(-40, 0), theta_d_step=1,
        theta_i_range=(0, 40), theta_i_step=1,
        fixed_interferers=[30.0],
        isr_values_db=[10.0],
        num_layers=3, hidden_nodes=256,
        num_samples=1_000_000, batch_size=1024, epochs=50,
    ),
    2: ScenarioConfig(
        name="Scenario 2 (N=8, Q=2, both variable)",
        N=8, Q=2,
        theta_d_range=(-40, 40), theta_d_step=5,
        theta_i_range=(-40, 40), theta_i_step=5,
        fixed_interferers=[],
        isr_values_db=[5.0, 10.0, 15.0, 20.0],
        num_layers=5, hidden_nodes=1024,
        num_samples=1_000_000, batch_size=1024, epochs=50,
    ),
    3: ScenarioConfig(
        name="Scenario 3 (N=8, Q=4, all variable)",
        N=8, Q=4,
        theta_d_range=(-40, 40), theta_d_step=5,
        theta_i_range=(-40, 40), theta_i_step=5,
        fixed_interferers=[],
        isr_values_db=[5.0, 10.0, 15.0, 20.0],
        num_layers=7, hidden_nodes=1024,
        num_samples=500_000, batch_size=1024, epochs=100,
    ),
    4: ScenarioConfig(
        name="Scenario 4 (N=16, Q=2, both variable)",
        N=16, Q=2,
        theta_d_range=(-40, 40), theta_d_step=5,
        theta_i_range=(-40, 40), theta_i_step=5,
        fixed_interferers=[],
        isr_values_db=[5.0, 10.0, 15.0, 20.0],
        num_layers=5, hidden_nodes=1024,
        num_samples=1_000_000, batch_size=1024, epochs=50,
    ),
}