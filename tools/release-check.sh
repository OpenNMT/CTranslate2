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
workdir=$(mktemp -d)
trap 'result=$?; rm -rf "$workdir"; publish "$state" || exit 1; exit "$result"' EXIT
artifact=python-wheels-Linux-auto64
[[ $backend == rocm ]] && artifact=rocm-python-wheels-Linux
runs=$(gh run list --repo "$repo" --workflow ci.yml --commit "$sha" --limit 100 --json databaseId --jq '.[].databaseId')
run=
for candidate in $runs; do
    artifacts=$(gh api --paginate "repos/$repo/actions/runs/$candidate/artifacts" --jq '.artifacts[] | select(.expired == false) | .name')
    if [[ $'\n'"$artifacts"$'\n' == *$'\n'"$artifact"$'\n'* ]]; then
        run=$candidate
        break
    fi
done
if [[ -z $run ]]; then
    echo "No CI run for commit $sha contains an available $artifact artifact." >&2
    exit 1
fi
echo "Downloading $artifact from https://github.com/$repo/actions/runs/$run"
gh run download "$run" --repo "$repo" --name "$artifact" --dir "$workdir/wheels"
python -m venv --system-site-packages "$workdir/venv"
python="$workdir/venv/bin/python"
"$python" -m pip install --no-index --no-deps --force-reinstall --find-links "$workdir/wheels" ctranslate2
"$python" -I -c "import ctranslate2; print(ctranslate2.__file__); assert ctranslate2.get_cuda_device_count() > 0, 'No GPU available in the PR wheel; check drivers and runtime libraries'"
test -d tests/data/models/transliteration-aren-all
tests/ctranslate2_test tests/data
"$python" -I -m pytest -v python/tests/ --ignore=python/tests/test_fairseq.py --ignore=python/tests/test_opennmt_py.py --ignore=python/tests/test_opennmt_tf.py
state=success
