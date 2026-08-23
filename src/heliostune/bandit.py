"""Bayesian linear Thompson sampling for kernel configuration selection."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Self, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from heliostune.features import FEATURE_NAMES

ActionT = TypeVar("ActionT")


class BayesianLinearBandit:
    """A Gaussian linear posterior with a zero-mean isotropic ridge prior.

    Observations follow ``y = x @ weights + error`` with independent error
    variance ``noise_variance``. The posterior is represented in information
    form so that each observation adds one rank-one precision update. Posterior
    means and samples use Cholesky solves; the covariance is never inverted or
    formed explicitly.
    """

    def __init__(
        self,
        dimension: int = len(FEATURE_NAMES),
        *,
        prior_precision: float = 1.0,
        noise_variance: float = 1.0,
        seed: int = 0,
    ) -> None:
        """Initialize an untouched ridge posterior and deterministic RNG.

        Args:
            dimension: Number of coefficients in every feature vector.
            prior_precision: Positive scalar precision of the zero-mean prior.
            noise_variance: Positive variance of each Gaussian observation.
            seed: Seed used by posterior sampling.

        Raises:
            TypeError: If ``dimension`` is not an integer.
            ValueError: If a dimension or distribution parameter is invalid.
        """
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._validate_positive_finite(prior_precision, "prior_precision")
        self._validate_positive_finite(noise_variance, "noise_variance")

        self._dimension = int(dimension)
        self._prior_precision = float(prior_precision)
        self._noise_variance = float(noise_variance)
        self._precision = np.eye(self._dimension, dtype=np.float64) * self._prior_precision
        self._information = np.zeros(self._dimension, dtype=np.float64)
        self._observed_count = 0
        self._cholesky: NDArray[np.float64] | None = None
        self._mean: NDArray[np.float64] | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def observed_count(self) -> int:
        """Return the number of observations added directly to this posterior."""
        return self._observed_count

    @property
    def mean(self) -> NDArray[np.float64]:
        """Return a copy of the posterior coefficient mean."""
        if self._mean is None:
            cholesky = self._posterior_cholesky()
            intermediate = np.linalg.solve(cholesky, self._information)
            self._mean = np.linalg.solve(cholesky.T, intermediate)
        return self._mean.copy()

    @property
    def precision(self) -> NDArray[np.float64]:
        """Return a copy of the posterior coefficient precision matrix."""
        return self._precision.copy()

    def update(self, features: ArrayLike, observation: float) -> None:
        """Condition the posterior on one feature vector and scalar observation.

        Args:
            features: One finite vector with length equal to ``dimension``.
            observation: A finite scalar response, normally a reward to maximize.

        Raises:
            ValueError: If either input has the wrong shape, is non-scalar, is
                non-finite, or would produce non-finite sufficient statistics.
        """
        vector = self._validated_features(features)
        try:
            response = float(observation)
        except (TypeError, ValueError) as exc:
            raise ValueError("observation must be a finite scalar") from exc
        if not math.isfinite(response):
            raise ValueError("observation must be a finite scalar")

        noise_precision = 1.0 / self._noise_variance
        with np.errstate(over="ignore", invalid="ignore"):
            precision_increment = noise_precision * np.outer(vector, vector)
            information_increment = noise_precision * vector * response
        if not np.all(np.isfinite(precision_increment)) or not np.all(
            np.isfinite(information_increment)
        ):
            raise ValueError("observation would produce non-finite sufficient statistics")

        self._precision += precision_increment
        self._information += information_increment
        self._observed_count += 1
        self._cholesky = None
        self._mean = None

    def sample(self) -> NDArray[np.float64]:
        """Draw and return one coefficient vector from the posterior."""
        cholesky = self._posterior_cholesky()
        standard_normal = self._rng.standard_normal(self._dimension)
        perturbation = np.linalg.solve(cholesky.T, standard_normal)
        return self.mean + perturbation

    def choose(
        self,
        actions: Sequence[ActionT],
        feature_fn: Callable[[ActionT], ArrayLike],
    ) -> ActionT:
        """Choose the highest-scoring action under one Thompson sample.

        A single coefficient vector is drawn for the entire action set. Ties are
        resolved deterministically in favor of the earliest action.

        Args:
            actions: Candidate actions in tie-breaking order.
            feature_fn: Function producing a feature vector for one action.

        Returns:
            The candidate maximizing sampled linear reward.

        Raises:
            ValueError: If ``actions`` is empty, generated features are invalid,
                or a sampled action score is non-finite.
        """
        if len(actions) == 0:
            raise ValueError("actions must not be empty")
        feature_rows = np.stack(
            [self._validated_features(feature_fn(action)) for action in actions]
        )
        coefficient_sample = self.sample()
        with np.errstate(over="ignore", invalid="ignore"):
            scores = feature_rows @ coefficient_sample
        if not np.all(np.isfinite(scores)):
            raise ValueError("action scores must be finite")
        return actions[int(np.argmax(scores))]

    def transferred(
        self,
        transfer_strength: float,
        *,
        prior_precision: float | None = None,
        noise_variance: float | None = None,
        seed: int = 0,
    ) -> Self:
        """Construct a target posterior from this discounted source likelihood.

        If the source precision and information vector are
        ``ridge_source * I + L`` and ``h``, respectively, this constructs
        ``ridge_target * I + strength * L`` and ``strength * h``. Thus the
        source prior is removed before discounting and the target ridge prior is
        retained. A strength of zero returns an untouched target prior; a
        strength of one and the default target ridge reproduce the source
        sufficient statistics. Transferred evidence does not increment the
        target's ``observed_count``; that count tracks target observations only.

        Args:
            transfer_strength: Finite likelihood power in the closed interval
                ``[0, 1]``.
            prior_precision: Target ridge precision, or this model's value when
                omitted.
            noise_variance: Variance for future target observations, or this
                model's value when omitted.
            seed: Seed used by target posterior sampling.

        Returns:
            A new posterior independent of this source posterior.

        Raises:
            ValueError: If the transfer strength or a target parameter is invalid.
        """
        try:
            strength = float(transfer_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError("transfer_strength must be finite and between 0 and 1") from exc
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("transfer_strength must be finite and between 0 and 1")

        target_prior = self._prior_precision if prior_precision is None else prior_precision
        target_noise = self._noise_variance if noise_variance is None else noise_variance
        target = type(self)(
            self._dimension,
            prior_precision=target_prior,
            noise_variance=target_noise,
            seed=seed,
        )
        if strength != 0.0:
            source_likelihood_precision = self._precision.copy()
            diagonal = np.diag_indices(self._dimension)
            source_likelihood_precision[diagonal] -= self._prior_precision
            target._precision += strength * source_likelihood_precision
            target._information = strength * self._information
        return target

    def _validated_features(self, features: ArrayLike) -> NDArray[np.float64]:
        """Return one detached float64 feature vector after validation."""
        try:
            vector = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("features must be a finite one-dimensional numeric vector") from exc
        if vector.shape != (self._dimension,):
            raise ValueError(f"features must have shape ({self._dimension},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("features must contain only finite values")
        return vector.copy()

    def _posterior_cholesky(self) -> NDArray[np.float64]:
        """Return the cached lower Cholesky factor of posterior precision."""
        if self._cholesky is None:
            self._cholesky = np.linalg.cholesky(self._precision)
        return self._cholesky

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> None:
        """Raise ``ValueError`` unless ``value`` is a positive finite scalar."""
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
