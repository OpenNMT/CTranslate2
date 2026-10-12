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
if git remote get-url upstream >/dev/null 2>&1; then
    repo=$(git remote get-url upstream)
else
    repo=$(git remote get-url origin)
fi
repo=${repo#*github.com[:/]}
repo=${repo%.git}
sha=$(git rev-parse HEAD)
branch=$(git branch --show-current)
pr=$(gh pr list --repo "$repo" --state open --head "$branch" --json number,headRefOid --jq ".[] | select(.headRefOid == \"$sha\") | .number" | head -n 1)
if [[ -z $pr ]]; then
    echo "Local HEAD is not the head commit of an open PR for $branch in $repo." >&2
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
run=${RELEASE_CHECK_RUN_ID:-}
if [[ -n $run ]]; then
    artifacts=$(gh api --paginate "repos/$repo/actions/runs/$run/artifacts" --jq '.artifacts[] | select(.expired == false) | .name')
    if [[ $'\n'"$artifacts"$'\n' != *$'\n'"$artifact"$'\n'* ]]; then
        echo "CI run $run does not contain an available $artifact artifact." >&2
        exit 1
    fi
else
    runs=$(gh run list --repo "$repo" --workflow ci.yml --commit "$sha" --limit 100 --json databaseId --jq '.[].databaseId')
    for candidate in $runs; do
        artifacts=$(gh api --paginate "repos/$repo/actions/runs/$candidate/artifacts" --jq '.artifacts[] | select(.expired == false) | .name')
        if [[ $'\n'"$artifacts"$'\n' == *$'\n'"$artifact"$'\n'* ]]; then
            run=$candidate
            break
        fi
    done
fi
if [[ -z $run ]]; then
    echo "No CI run for commit $sha contains an available $artifact artifact." >&2
    exit 1
fi
echo "Downloading $artifact from https://github.com/$repo/actions/runs/$run"
gh run download "$run" --repo "$repo" --name "$artifact" --dir "$workdir/wheels"
python -m venv "$workdir/venv"
python="$workdir/venv/bin/python"
"$python" -m pip install -r python/tests/requirements.txt
"$python" -m pip install --force-reinstall --find-links "$workdir/wheels" ctranslate2
"$python" -I -c "import ctranslate2; print(ctranslate2.__file__); assert ctranslate2.get_cuda_device_count() > 0, 'No GPU available in the PR wheel; check drivers and runtime libraries'"
if [[ -x tests/ctranslate2_test ]]; then
    tests/ctranslate2_test tests/data
else
    echo "Skipping native tests: tests/ctranslate2_test is not built." >&2
fi
"$python" -I -m pytest -v python/tests/ --ignore=python/tests/test_fairseq.py --ignore=python/tests/test_opennmt_py.py --ignore=python/tests/test_opennmt_tf.py
state=success
