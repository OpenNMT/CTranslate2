#!/usr/bin/env bash
# Usage: bash release-check.sh cuda|rocm PR_NUMBER_OR_URL COMMAND [ARG...]
set -euo pipefail
if (( $# < 3 )) || [[ $1 != cuda && $1 != rocm ]]; then
    echo "Usage: bash release-check.sh cuda|rocm PR_NUMBER_OR_URL COMMAND [ARG...]" >&2
    exit 2
fi
backend=$1
pr=$2
shift 2
cd "$(git rev-parse --show-toplevel)"

if ! git diff HEAD --quiet; then
    echo "Commit or stash changes to tracked files before validation." >&2
    exit 1
fi
IFS=$'\t' read -r sha url < <(
    gh pr view "$pr" --json headRefOid,url --jq '[.headRefOid, .url] | @tsv'
)
if [[ $(git rev-parse HEAD) != "$sha" ]]; then
    echo "Local HEAD must match the PR's latest commit." >&2
    exit 1
fi
repo=${url#https://github.com/}
repo=${repo%/pull/*}
publish() {
    gh api "repos/$repo/statuses/$sha" \
        -f "state=$1" -f "context=release/$backend" >/dev/null
}
publish pending
state=failure
trap 'result=$?; publish "$state" || exit 1; exit "$result"' EXIT
"$@"
state=success
