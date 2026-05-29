# tests/test_market_env.py
import numpy as np
import pytest
from src.env.market_env import MarketEnv, OBS_DIM, CATEGORIES

@pytest.fixture
def env():
    return MarketEnv(seed=42)

def test_obs_dim_is_9():
    assert OBS_DIM == 9

def test_reset_returns_correct_shape(env):
    obs = env.reset()
    for cat in CATEGORIES:
        assert obs[cat].shape == (OBS_DIM,)

def test_comp_ratio_in_obs(env):
    obs = env.reset()
    # obs[8] = competitor_ratio, should be in [0.5, 2.0]
    for cat in CATEGORIES:
        assert 0.5 <= obs[cat][8] <= 2.0

def test_step_returns_four_categories(env):
    env.reset()
    deltas = {cat: 0.0 for cat in CATEGORIES}
    obs, info, done, _ = env.step(deltas)
    assert set(obs.keys()) == set(CATEGORIES)

def test_waste_fires_at_050(env):
    """Force freshness below 0.50, check inventory cleared."""
    env.reset()
    # Manually set freshness of leafy to 0.49
    env._freshness["leafy"] = 0.49
    env._inventory["leafy"] = 10
    deltas = {cat: 0.0 for cat in CATEGORIES}
    _, info, _, _ = env.step(deltas)
    assert env._inventory["leafy"] == 0

def test_restock_rejected_below_070(env):
    """If f_delivery < 0.70, inventory should not increase."""
    env.reset()
    env._inventory["herbs"] = 5
    env._freshness["herbs"] = 0.80
    initial_inv = env._inventory["herbs"]
    env._attempt_restock("herbs", f_delivery=0.65)
    assert env._inventory["herbs"] == initial_inv  # rejected

def test_price_never_below_cost_floor(env):
    env.reset()
    for _ in range(30):
        deltas = {cat: -0.30 for cat in CATEGORIES}
        env.step(deltas)
    for cat in CATEGORIES:
        cost_floor = env._demand_model._params[cat]["ref_price"] * \
                     env._demand_model._params[cat]["cost_ratio"] * 1.05
        assert env._prices[cat] >= cost_floor - 1e-6

def test_prev_delta_in_obs(env):
    env.reset()
    deltas = {cat: -0.10 for cat in CATEGORIES}
    obs, _, _, _ = env.step(deltas)
    # obs[7] = prev_delta, should be -0.10
    for cat in CATEGORIES:
        assert obs[cat][7] == pytest.approx(-0.10, abs=1e-4)
