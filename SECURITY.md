# Security policy

## Supported versions

I support base and CPU-only functionality in the latest
[GitHub release](https://github.com/mottopanikeiku/quartet/releases/latest).
Earlier releases are unsupported. I evaluate reports and issue fixes when
warranted, without a guaranteed response time.

Historical GPU dependency pins are for reproducing experiments. Use them only
with trusted inputs in an isolated environment, not in a service or a
multi-tenant deployment. Security fixes ship in new versions; they do not
rewrite benchmark measurements or reports.

## Offline analyzer replay

`heliostune replay-bundle` runs only installed analyzers from a fixed registry.
A bundle cannot choose a command, module, path or callable to execute.
Verification limits reads to 2 MiB for a bundle root, 32 MiB per component and
64 MiB in total. The worker checks that its installed package and sources match
the parent's identity before running.

Replay requires Linux user, network, mount and PID namespaces with
`no-new-privileges`. Each worker checks its namespace mappings, mounts an empty
`nosuid,nodev,noexec` tmpfs, remounts it read-only, checks `ST_RDONLY` and an
`EROFS` write failure, then chroots into it. Resource limits, closed non-stdio
descriptors, bounded output and process-group termination constrain failures.
The Python audit hook adds detection of forbidden operations; it is not a
general Python sandbox. Replay fails if required isolation is unavailable.

The analyzer runs twice and must reproduce the bundle's declared outputs
exactly. This checks same-host output reproduction, not the accuracy of GPU
measurements or their reproducibility elsewhere. The isolation mechanisms do
not make malicious installed packages safe. See the
[bundle and replay format](docs/history/EXPERIMENT_SCOPE.md#offline-analyzer-replay)
for input and output contracts.

## Reporting a vulnerability

Use GitHub's
[private reporting form](https://github.com/mottopanikeiku/quartet/security/advisories/new)
for vulnerabilities, exposed credentials or release-integrity concerns.
Include affected versions, impact and reproduction details. Please report an
unaddressed vulnerability privately before publishing it.
