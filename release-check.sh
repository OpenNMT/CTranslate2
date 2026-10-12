#!/usr/bin/env bash
# Usage: bash release-check.sh cuda|rocm
set -euo pipefail
if (( $# != 1 )) || [[ $1 != cuda && $1 != rocm ]]; then
    echo "Usage: bash release-check.sh cuda|rocm" >&2
    exit 2
fi
backend=$1
cd "$(git rev-parse --show-toplevel)"

if ! git diff HEAD --quiet; then
    echo "Commit or stash changes to tracked files before validation." >&2
    exit 1
fi
IFS=$'\t' read -r sha url < <(
    gh pr view --json headRefOid,url --jq '[.headRefOid, .url] | @tsv'
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
tests/ctranslate2_test tests/data
python -m pytest -v python/tests/ --ignore=python/tests/test_fairseq.py --ignore=python/tests/test_opennmt_py.py --ignore=python/tests/test_opennmt_tf.py
state=success
