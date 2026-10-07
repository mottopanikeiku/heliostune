# Security policy

I support base and CPU-only functionality in the latest [GitHub release](https://github.com/mottopanikeiku/heliostune/releases/latest), not earlier releases. I evaluate security reports and issue fixes when warranted. This experiment is not a security boundary, and I cannot guarantee a response time.

I retain historical GPU dependency pins for reproduction, not as supported service dependencies. They should run only with trusted inputs in an isolated environment. Fixes belong in new code, not rewritten benchmark measurements.

`heliostune replay-bundle` runs package-shipped analyzers from a fixed registry. It requires Linux isolation features and rejects a run when they are unavailable. Bounded inputs, namespace isolation and resource limits do not make malicious installed packages safe. A successful replay reproduces declared output bytes; it does not establish that the measurements are true or reproducible on another GPU. I retain the implementation boundaries in the [historical security policy](docs/history/SECURITY.md).

I accept vulnerability reports through GitHub's [private reporting form](https://github.com/mottopanikeiku/heliostune/security/advisories/new). Include affected versions, impact and reproduction details. Please report an unaddressed vulnerability privately before publishing it.
