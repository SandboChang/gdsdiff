# Security Policy

Please report suspected vulnerabilities privately through GitHub's security
advisory interface for this repository. Do not include sensitive layout data in
an issue. The pre-release `0.1.x` line receives security fixes on its current
development branch.

The optional geometry cache never deserializes Python objects and ignores
legacy pickle entries. Native cache entries are executable libraries, so keep
the default private user cache. A custom `GDSDIFF_NATIVE_CACHE_DIR` is a trusted
boundary and must not be shared with untrusted writers.
