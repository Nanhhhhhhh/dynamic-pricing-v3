import numpy as np
import pytest
from src.mpc.controller import MPC, MPCConfig
from src.env.market_env import OBS_DIM, OBS_WINDOW

def _make_obs_window(freshness, inv_ratio, prev_delta=0.0):
    """Construct a (21, 9) obs window with constant state."""
    row = np.zeros(OBS_DIM, dtype=np.float32)
    row[0] = freshness
    row[1] = inv_ratio
    row[2] = 1.0
    row[7] = prev_delta
    row[8] = 1.0
    return np.tile(row, (OBS_WINDOW, 1))

@pytest.fixture
def mpc():
    return MPC(MPCConfig())

def test_high_freshness_does_not_max_discount(mpc):
    """At f=0.9, delta should not be -0.30 (move penalty should suppress it)."""
    obs = _make_obs_window(freshness=0.90, inv_ratio=0.5)
    result = mpc.decide(obs, category="leafy", current_price=1.48,
                         current_inv=50, current_freshness=0.90, prev_delta=0.0)
    assert result["delta"] > -0.30, f"Expected non-max-discount at f=0.9, got {result['delta']}"

def test_low_freshness_high_inv_discounts(mpc):
    """At f=0.35 with high inventory, delta should be negative."""
    obs = _make_obs_window(freshness=0.35, inv_ratio=1.5)
    result = mpc.decide(obs, category="leafy", current_price=1.48,
                         current_inv=150, current_freshness=0.35, prev_delta=0.0)
    assert result["delta"] < 0.0, f"Expected discount at f=0.35 high inv, got {result['delta']}"

def test_result_has_required_keys(mpc):
    obs = _make_obs_window(freshness=0.8, inv_ratio=0.5)
    result = mpc.decide(obs, "leafy", 1.48, 50, 0.8, 0.0)
    for key in ("delta", "scores", "d_hat_0", "p_waste_0", "reason"):
        assert key in result, f"Missing key: {key}"

def test_delta_within_bounds(mpc):
    obs = _make_obs_window(freshness=0.7, inv_ratio=0.8)
    result = mpc.decide(obs, "herbs", 4.54, 30, 0.7, 0.0)
    assert -0.30 <= result["delta"] <= 0.20

def test_clearability_override_for_root(mpc):
    """Root (inelastic, β=-0.457) with imminent waste should get max discount via override."""
    # f=0.52 is just above threshold, shelf_life ≈ (log(0.50/0.52)/log(0.95)) ≈ 0.76 days
    # t_critical < clearability_horizon=6, large inventory that can't be cleared
    obs = _make_obs_window(freshness=0.52, inv_ratio=2.0)
    result = mpc.decide(obs, category="root", current_price=1.06,
                         current_inv=200, current_freshness=0.52, prev_delta=0.0)
    assert result["delta"] == pytest.approx(-0.30, abs=0.01)
