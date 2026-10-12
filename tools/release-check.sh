#!/usr/bin/env bash
# Usage: bash tools/release-check.sh cuda|rocm
set -euo pipefail
if (( $# != 1 )) || [[ $1 != cuda && $1 != rocm ]]; then
    echo "Usage: bash tools/release-check.sh cuda|rocm" >&2
    exit 2
fi
backend=$1
cd "$(git rev-parse --show-toplevel)"

if ! git diff HEAD --quiet; then
    echo "Commit or stash changes to tracked files before validation." >&2
    exit 1
fi
repo=$(git remote get-url origin)
repo=${repo#*github.com[:/]}
repo=${repo%.git}
sha=$(gh pr view "$(git branch --show-current)" --repo "$repo" --json headRefOid --jq .headRefOid)
if [[ $(git rev-parse HEAD) != "$sha" ]]; then
    echo "Local HEAD must match the PR's latest commit." >&2
    exit 1
fi
publish() {
    gh api "repos/$repo/statuses/$sha" \
        -f "state=$1" -f "context=release/$backend" >/dev/null
}
publish pending
state=failure
trap 'result=$?; publish "$state" || exit 1; exit "$result"' EXIT
python -c "import ctranslate2; assert ctranslate2.get_cuda_device_count() > 0, 'No GPU available'"
tests/ctranslate2_test tests/data
python -m pytest -v python/tests/ --ignore=python/tests/test_fairseq.py --ignore=python/tests/test_opennmt_py.py --ignore=python/tests/test_opennmt_tf.py
state=success
