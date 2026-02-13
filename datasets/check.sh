#!/usr/bin/env bash
# Verify dataset files via checksums.
# Exit 0 = valid (skip download), Exit 1 = need download.
#
# Usage: ../check.sh (from a dataset directory with checksums.sha256)

set -euo pipefail

if [ ! -f checksums.sha256 ]; then
    exit 1
fi

if sha256sum -c checksums.sha256 --quiet 2>/dev/null; then
    echo "Checksums OK, skipping download."
    exit 0
fi

exit 1
