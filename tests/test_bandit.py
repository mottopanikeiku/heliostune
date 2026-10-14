import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

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


def test_samples_follow_the_posterior_mean_and_covariance() -> None:
    model = BayesianLinearBandit(dimension=2, prior_precision=0.5, noise_variance=0.25, seed=11)
    model.update([1.0, 2.0], 1.5)
    model.update([-1.0, 0.5], -0.5)
    model.update([0.3, -1.0], 0.2)
    covariance = np.linalg.inv(model.precision)

    samples = np.stack([model.sample() for _ in range(40_000)])

    np.testing.assert_allclose(samples.mean(axis=0), model.mean, atol=0.01)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), covariance, rtol=0.05, atol=1e-3)


def test_choose_breaks_exact_score_ties_in_favor_of_the_earliest_action() -> None:
    model = BayesianLinearBandit(dimension=2, seed=5)
    actions = ("first", "second", "third")
    features = {"first": [1.0, 1.0], "second": [1.0, 1.0], "third": [1.0, 1.0]}
    assert model.choose(actions, features.__getitem__) == "first"


@pytest.mark.parametrize("dimension", [0, -1])
def test_constructor_rejects_nonpositive_dimensions(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        BayesianLinearBandit(dimension=dimension)


@pytest.mark.parametrize("dimension", [True, 2.0, "2"])
def test_constructor_rejects_noninteger_dimensions(dimension: object) -> None:
    with pytest.raises(TypeError, match="dimension must be an integer"):
        BayesianLinearBandit(dimension=dimension)  # type: ignore[arg-type]


@pytest.mark.parametrize("parameter", ["prior_precision", "noise_variance"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), "x"])
def test_constructor_rejects_invalid_distribution_parameters(parameter: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be positive and finite"):
        BayesianLinearBandit(dimension=1, **{parameter: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("observation", [float("nan"), float("inf"), "x", None, [1.0, 2.0]])
def test_update_rejects_nonfinite_or_nonscalar_observations(observation: object) -> None:
    model = BayesianLinearBandit(dimension=1)
    with pytest.raises(ValueError, match="finite scalar"):
        model.update([1.0], observation)  # type: ignore[arg-type]
    assert model.observed_count == 0


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ([1.0], r"shape \(2,\), got \(1,\)"),
        ([[1.0, 2.0]], r"shape \(2,\), got \(1, 2\)"),
        ([1.0, float("nan")], "only finite values"),
        (["a", "b"], "finite one-dimensional numeric vector"),
    ],
)
def test_invalid_feature_vectors_are_rejected(features: object, message: str) -> None:
    model = BayesianLinearBandit(dimension=2)
    with pytest.raises(ValueError, match=message):
        model.update(features, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        model.choose(["only"], lambda _action: features)  # type: ignore[arg-type,return-value]
    assert model.observed_count == 0


def test_choose_rejects_empty_actions_and_overflowing_scores() -> None:
    with pytest.raises(ValueError, match="actions must not be empty"):
        BayesianLinearBandit(dimension=1).choose([], lambda action: [1.0])
    # A near-zero prior precision makes posterior samples enormous.
    diffuse = BayesianLinearBandit(dimension=1, prior_precision=1e-300, seed=1)
    with pytest.raises(ValueError, match="action scores must be finite"):
        diffuse.choose(["huge"], lambda _action: [1e308])


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf"), "x", None])
def test_transfer_strength_must_be_in_the_unit_interval(strength: object) -> None:
    with pytest.raises(ValueError, match="transfer_strength must be finite and between 0 and 1"):
        BayesianLinearBandit(dimension=1).transferred(strength)  # type: ignore[arg-type]


_UpdateBatch = tuple[int, list[list[float]], list[float]]

_BOUNDED_FLOATS = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)
_POSITIVE_SCALES = st.floats(
    min_value=0.1,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _update_batches(draw: st.DrawFn) -> _UpdateBatch:
    """Draw a dimension plus a matching run of feature/observation pairs."""
    dimension = draw(st.integers(min_value=1, max_value=5))
    count = draw(st.integers(min_value=0, max_value=12))
    features = draw(
        st.lists(
            st.lists(_BOUNDED_FLOATS, min_size=dimension, max_size=dimension),
            min_size=count,
            max_size=count,
        )
    )
    observations = draw(st.lists(_BOUNDED_FLOATS, min_size=count, max_size=count))
    return dimension, features, observations


@st.composite
def _choice_problems(draw: st.DrawFn) -> tuple[_UpdateBatch, list[list[float]], int]:
    """Draw a conditioned posterior together with a nonempty action feature table."""
    batch = draw(_update_batches())
    dimension = batch[0]
    action_features = draw(
        st.lists(
            st.lists(_BOUNDED_FLOATS, min_size=dimension, max_size=dimension),
            min_size=1,
            max_size=6,
        )
    )
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return batch, action_features, seed


def _fitted(
    batch: _UpdateBatch,
    *,
    prior_precision: float = 1.0,
    noise_variance: float = 1.0,
    seed: int = 0,
) -> BayesianLinearBandit:
    dimension, features, observations = batch
    model = BayesianLinearBandit(
        dimension=dimension,
        prior_precision=prior_precision,
        noise_variance=noise_variance,
        seed=seed,
    )
    for vector, observation in zip(features, observations, strict=True):
        model.update(vector, observation)
    return model


@given(
    batch=_update_batches(),
    prior_precision=_POSITIVE_SCALES,
    noise_variance=_POSITIVE_SCALES,
)
def test_precision_is_the_ridge_prior_plus_scaled_gram_matrix(
    batch: _UpdateBatch,
    prior_precision: float,
    noise_variance: float,
) -> None:
    dimension, features, _ = batch
    model = _fitted(batch, prior_precision=prior_precision, noise_variance=noise_variance)
    rows = np.asarray(features, dtype=np.float64).reshape(len(features), dimension)
    expected = np.eye(dimension) * prior_precision + rows.T @ rows / noise_variance

    precision = model.precision

    np.testing.assert_allclose(precision, expected, rtol=1e-9, atol=1e-6)
    np.testing.assert_array_equal(precision, precision.T)
    assert np.all(np.isfinite(np.linalg.cholesky(precision)))


@given(batch=_update_batches())
def test_observed_count_equals_the_number_of_update_calls(batch: _UpdateBatch) -> None:
    dimension, features, observations = batch
    model = BayesianLinearBandit(dimension=dimension)
    assert model.observed_count == 0

    pairs = enumerate(zip(features, observations, strict=True), start=1)
    for call_number, (vector, observation) in pairs:
        model.update(vector, observation)
        assert model.observed_count == call_number

    assert model.observed_count == len(features)


@given(batch=_update_batches())
def test_mean_and_precision_hand_out_detached_copies(batch: _UpdateBatch) -> None:
    model = _fitted(batch)
    precision = model.precision
    mean = model.mean
    assert precision is not model.precision
    assert mean is not model.mean

    precision_snapshot = precision.copy()
    mean_snapshot = mean.copy()
    precision[...] = np.nan
    mean[...] = np.nan

    np.testing.assert_array_equal(model.precision, precision_snapshot)
    np.testing.assert_array_equal(model.mean, mean_snapshot)


@given(problem=_choice_problems())
def test_choose_returns_one_of_the_supplied_actions(
    problem: tuple[_UpdateBatch, list[list[float]], int],
) -> None:
    batch, action_features, seed = problem
    model = _fitted(batch, seed=seed)
    actions = tuple(f"action-{index}" for index in range(len(action_features)))
    features = dict(zip(actions, action_features, strict=True))

    assert model.choose(actions, features.__getitem__) in actions
    # Restricting the candidate set must restrict the outcome: no action outside
    # the supplied sequence can ever be returned.
    assert model.choose(actions[:1], features.__getitem__) == actions[0]
