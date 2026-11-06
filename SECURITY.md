# Security policy

## Supported versions

Security fixes go into the latest release on [PyPI](https://pypi.org/project/quantlab-py/).
While the version is 0.x, older releases are not patched.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/ZhaorongDai/quantlab/security/advisories/new), with the affected
version, the steps to reproduce it and the impact you expect.

A report that an API key or other credential can leak is in scope. quantlab reads credentials
from environment variables only, so a key found in the repository or written to a store, log or
report is a vulnerability.
