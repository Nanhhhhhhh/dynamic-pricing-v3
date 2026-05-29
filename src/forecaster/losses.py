import torch
import torch.nn.functional as F


def pos_weight_from_rate(positive_rate: float) -> float:
    return (1.0 - positive_rate) / positive_rate


def combined_loss(
    demand_pred: torch.Tensor,
    demand_true: torch.Tensor,
    waste_logit: torch.Tensor,
    waste_true: torch.Tensor,
    pos_weight: float,
    w_demand: float = 1.0,
    w_waste: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor(pos_weight, dtype=torch.float32)
    l_demand = F.huber_loss(demand_pred, demand_true, delta=1.0)
    l_waste  = F.binary_cross_entropy_with_logits(
        waste_logit, waste_true, pos_weight=pw
    )
    return w_demand * l_demand + w_waste * l_waste
