#!/usr/bin/env bash
# The study coordinator does no application compilation on the login node.
set -euo pipefail
study_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$study_source/scripts/perf/gpu_study.py" "$@"
