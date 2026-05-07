#!/usr/bin/env bash
# pre-push hook + CI step: audit uv-resolved deps for known CVEs.
# Reads .pip-audit-ignore (one CVE/GHSA ID per line, # comments OK).
# Exit non-zero on any unignored vulnerability so push blocks.
#
# Audits the uv.lock-resolved set (not the live venv) so results are
# deterministic and independent of the developer's local sync state.
set -euo pipefail

PIP_AUDIT_VERSION="${PIP_AUDIT_VERSION:-2.10.0}"
IGNORE_FILE=".pip-audit-ignore"

if [[ ! -f uv.lock ]]; then
  echo "[pip-audit] no uv.lock in $(pwd) — nothing to audit" >&2
  exit 0
fi

ignore_args=()
if [[ -f "$IGNORE_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | tr -d '[:space:]')"
    [[ -z "$line" ]] && continue
    ignore_args+=(--ignore-vuln "$line")
  done < "$IGNORE_FILE"
fi

uv export --frozen --no-emit-project --all-extras --all-groups \
  --format requirements-txt --no-hashes --quiet \
| uvx --quiet --from "pip-audit==${PIP_AUDIT_VERSION}" \
    pip-audit -r /dev/stdin --disable-pip --no-deps "${ignore_args[@]}"
