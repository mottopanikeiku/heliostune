"""Shared pytest configuration.

The Hypothesis profile is derandomized so that a property failure reported by
CI is reproducible from the same commit without a stored example database.
"""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "heliostune",
    derandomize=True,
    deadline=None,
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("heliostune")
