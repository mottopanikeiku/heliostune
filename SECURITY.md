# Security policy

The latest [GitHub release](https://github.com/mottopanikeiku/heliostune/releases/latest) is supported for base and CPU-only functionality. Earlier releases are not supported. Security support means evaluating reports and issuing a new release when warranted; it does not guarantee response time or make this experiment suitable for a security boundary.

Historical GPU dependency pins are reproduction inputs, not supported service dependencies. Use them only with trusted inputs in an isolated environment. Published benchmark files are retained unchanged; fixes ship as new code rather than rewriting old measurements.

`heliostune replay-bundle` runs only package-shipped analyzers selected from a fixed registry. It requires Linux isolation features and rejects a run when they are unavailable. Bounded inputs, namespace isolation and resource limits do not make installed malicious packages safe. A successful local replay demonstrates reproduction of declared output bytes, not the truth of the measurements or reproduction on another GPU. Detailed implementation boundaries are retained in the [historical security policy](docs/history/SECURITY.md).

Report vulnerabilities through GitHub's [private reporting form](https://github.com/mottopanikeiku/heliostune/security/advisories/new), including affected versions, impact and reproduction details. Do not publish an unaddressed vulnerability before private reporting.
