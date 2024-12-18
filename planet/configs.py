from dataclasses import dataclass, asdict
from typing import Optional, Dict


@dataclass
class TrainConfig:
    seed: int = 0
    log_dir: str = 'log'
    test_interval: int = 10
    action_repeat: int = 1
    state_dim: int = 30
    rnn_hidden_dim: int = 200
    obs_embedding_dim: int = 64
    buffer_capacity: int = 1000000
    all_episodes: int = 1000
    seed_episodes: int = 5
    collect_interval: int = 100
    batch_size: int = 50
    chunk_length: int = 50
    lr: float = 1e-3
    eps: float = 1e-5
    clip_grad_norm: int = 1000
    free_nats: int = 3
    horizon: int = 12
    N_iterations: int = 10
    N_candidates: int = 1000
    N_top_candidates: int = 100
    action_noise_var: float = 0.3

    dict = asdict


@dataclass
class TestConfig:
    model_dir: str
    render: bool = False
    action_repeat: int = 1
    episodes: int = 1
    horizon: int = 12
    N_iterations: int = 10
    N_candidates: int = 1000
    N_top_candidates: int = 100
    env_options: Optional[Dict] = None
    env_seed: Optional[int] = None

    dict = asdict