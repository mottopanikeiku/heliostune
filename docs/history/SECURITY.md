# Security Policy

## Supported versions

The latest [GitHub release](https://github.com/mottopanikeiku/heliostune/releases/latest) is supported for HeliosTune's base and CPU-only functionality. Earlier releases are not supported. Support is limited to evaluating security reports that affect maintained functionality and issuing fixes in a new version when a fix is warranted; it does not guarantee a response time or make research workloads suitable for a security boundary.

## Frozen GPU reproduction software and evidence

The GPU dependency pins and configurations preserved for reproduction are frozen research inputs, not maintained production dependencies. Use them only with trusted inputs in an isolated environment. They are not supported for services, multi-tenant use, or operation at a security boundary.

Published benchmark artifacts, reports, protocols, and other frozen evidence are never rewritten in response to a security fix. Fixes ship in new versions, and any new investigation or changed evidence uses a new protocol and path while preserving the original bytes.

## Offline analyzer replay boundary

`heliostune replay-bundle` executes only package-shipped, source-bound analyzer
IDs from a closed registry; bundle bytes cannot select a path, module, command,
entrypoint, or callable. Bundle roots, components, and aggregate captures have
finite pre-read limits. The worker request binds the selected implementation
and the parent's package/version/source identity, and the child independently
recaptures that identity before running.

Replay is Linux-specific and fails closed without the fixed no-new-privileges
user, network, mount, and PID namespace path. Each worker verifies its namespace
and user mappings, mounts an empty `nosuid,nodev,noexec` tmpfs, bind-remounts it
read-only, verifies `ST_RDONLY` plus an `EROFS` write probe, and chroots before
invoking the analyzer. Resource limits, closed non-stdio descriptors, bounded
regular-file output, and process-group kill/reap handling constrain failures.
The Python audit hook is only an additional deny-and-latch tripwire; it is not a
general Python sandbox.

A successful two-run drill establishes only same-host reproduction of the
bundle's declared committed derived bytes by that registered analyzer. It does
not authenticate the evidence, validate its semantics or statistics, reproduce
GPU measurements or a complete campaign, establish cross-host determinism, or
make HeliosTune suitable for executing malicious installed packages at a
security boundary.

## Private vulnerability reporting

Report vulnerabilities in the supported release, exposed project credentials, or repository and release-integrity concerns through GitHub's [private vulnerability reporting form](https://github.com/mottopanikeiku/heliostune/security/advisories/new). Include affected versions, impact, and reproduction details when possible. Please do not disclose an unaddressed vulnerability publicly.
