# Security Policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Do not open a public issue for a vulnerability.

Use GitHub's **Report a vulnerability** action in the repository Security tab
to submit a private security advisory. Include:

- Affected version or commit.
- Reproduction steps or a proof of concept.
- Expected impact.
- Any suggested mitigation.

You should receive an acknowledgement within seven days. Please allow time for
a fix and coordinated disclosure before publishing details.

## Scope

AttnLab is an educational local execution environment. It does not provide
authentication, authorization, durable session storage, or multi-tenant
isolation. Public deployments should be treated as demonstration instances and
must not process secrets or untrusted private data.
