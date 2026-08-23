import numpy as np

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
