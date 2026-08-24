import numpy as np
import pytest

from heliostune.bandit import BayesianLinearBandit


def test_power_prior_transfer_preserves_only_discounted_likelihood() -> None:
    source = BayesianLinearBandit(
        dimension=3,
        prior_precision=2.0,
        noise_variance=0.5,
        seed=3,
    )
    source.update([1.0, 0.0, 0.5], 2.0)
    source.update([0.0, 1.0, -0.5], -1.0)

    zero = source.transferred(0.0, seed=4)
    np.testing.assert_allclose(zero.precision, np.eye(3) * 2.0)
    np.testing.assert_allclose(zero.mean, np.zeros(3))
    assert zero.observed_count == 0

    full = source.transferred(1.0, seed=4)
    np.testing.assert_allclose(full.precision, source.precision)
    np.testing.assert_allclose(full.mean, source.mean)
    assert full.observed_count == 0


def test_thompson_choice_uses_one_valid_action() -> None:
    model = BayesianLinearBandit(dimension=2, seed=8)
    actions = ("small", "large")
    features = {"small": np.array([1.0, 0.0]), "large": np.array([0.0, 1.0])}
    assert model.choose(actions, features.__getitem__) in actions


def test_rejected_update_is_atomic_when_cholesky_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = BayesianLinearBandit(dimension=2)
    model.update([1.0, 0.0], 1.0)
    _ = model.mean
    precision = model.precision
    mean = model.mean
    cholesky = model._cholesky.copy()
    observed_count = model.observed_count

    def fail_cholesky(_matrix: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "cholesky", fail_cholesky)
    with pytest.raises(ValueError, match=r"dimension 2.*after 2 observations"):
        model.update([0.0, 1.0], 2.0)

    np.testing.assert_array_equal(model.precision, precision)
    np.testing.assert_array_equal(model._mean, mean)
    np.testing.assert_array_equal(model._cholesky, cholesky)
    assert model.observed_count == observed_count


def test_update_validates_cumulative_finiteness_before_commit() -> None:
    model = BayesianLinearBandit(dimension=1)
    model._precision[0, 0] = np.finfo(np.float64).max
    precision = model.precision

    with pytest.raises(ValueError, match="non-finite posterior state"):
        model.update([np.sqrt(np.finfo(np.float64).max)], 1.0)

    np.testing.assert_array_equal(model.precision, precision)
    assert model.observed_count == 0


def test_update_rejects_boolean_observations() -> None:
    model = BayesianLinearBandit(dimension=1)
    with pytest.raises(ValueError, match="finite scalar"):
        model.update([1.0], True)
