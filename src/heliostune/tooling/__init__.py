"""Report rendering, artifact IO, and experiment verification utilities.

The tuner algorithms remain in :mod:`heliostune.bandit`, :mod:`heliostune.retrieval`,
:mod:`heliostune.replay`, :mod:`heliostune.multisource_engine`, and
:mod:`heliostune.v3_engine`. Import tooling from its individual modules; importing
this package does not load report renderers, verifiers, or subprocess workers.
"""
