import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


@dataclass
class ForecasterConfig:
    obs_dim:       int   = 9
    window:        int   = 21
    n_categories:  int   = 4
    cat_embed_dim: int   = 8
    lstm_hidden:   int   = 128
    lstm_layers:   int   = 2
    lstm_dropout:  float = 0.2


class ForecasterLSTM(nn.Module):
    def __init__(self, cfg: ForecasterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cat_embed = nn.Embedding(cfg.n_categories, cfg.cat_embed_dim)
        self.lstm = nn.LSTM(
            input_size=cfg.obs_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )
        z_dim = cfg.lstm_hidden + cfg.cat_embed_dim
        self.demand_head = nn.Linear(z_dim, 1)
        self.waste_head  = nn.Linear(z_dim, 1)

    def forward(
        self, features: torch.Tensor, category_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # features: (B, T, obs_dim), category_idx: (B,)
        lstm_out, _ = self.lstm(features)
        last = lstm_out[:, -1, :]                          # (B, lstm_hidden)
        cat_vec = self.cat_embed(category_idx)             # (B, cat_embed_dim)
        z = torch.cat([last, cat_vec], dim=-1)             # (B, z_dim)
        return {
            "demand":      self.demand_head(z).squeeze(-1),
            "waste_logit": self.waste_head(z).squeeze(-1),
        }
