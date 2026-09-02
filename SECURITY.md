# Security Policy

## Supported versions

The latest [GitHub release](https://github.com/mottopanikeiku/heliostune/releases/latest) is supported for HeliosTune's base and CPU-only functionality. Earlier releases are not supported. Support is limited to evaluating security reports that affect maintained functionality and issuing fixes in a new version when a fix is warranted; it does not guarantee a response time or make research workloads suitable for a security boundary.

## Frozen GPU reproduction software and evidence

The GPU dependency pins and configurations preserved for reproduction are frozen research inputs, not maintained production dependencies. Use them only with trusted inputs in an isolated environment. They are not supported for services, multi-tenant use, or operation at a security boundary.

Published benchmark artifacts, reports, protocols, and other frozen evidence are never rewritten in response to a security fix. Fixes ship in new versions, and any new investigation or changed evidence uses a new protocol and path while preserving the original bytes.

## Private vulnerability reporting

Report vulnerabilities in the supported release, exposed project credentials, or repository and release-integrity concerns through GitHub's [private vulnerability reporting form](https://github.com/mottopanikeiku/heliostune/security/advisories/new). Include affected versions, impact, and reproduction details when possible. Please do not disclose an unaddressed vulnerability publicly.
