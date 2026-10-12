# Release validation

1. Open a release PR and wait for CI to pass.
2. On each GPU machine, check out the PR head and build/install its candidate
   wheel and native tests. Authenticate with `gh auth login` using an account
   permitted to write commit statuses.
3. Run the native and Python tests, including GPU tests:

   ```bash
   bash release-check.sh cuda PR_URL bash -ec '
     tests/ctranslate2_test tests/data
     python -m pytest -v python/tests/ --ignore=python/tests/test_fairseq.py --ignore=python/tests/test_opennmt_py.py --ignore=python/tests/test_opennmt_tf.py
   '
   ```

   Replace `cuda` with `rocm` on the AMD machine.

The script checks that tracked files are clean and HEAD matches the PR. It
publishes `release/cuda` or `release/rocm` as pending, then success or failure.
The backend argument only labels the status; configure GPU execution yourself.
The status succeeds only if both test commands pass. Ensure GPU tests run;
skipped GPU tests do not validate the backend. Detailed output stays in the terminal.

Configure both statuses as required checks for release PRs. Re-run validation
for every new commit. Publish the validated artifacts after CI and both GPU
checks pass; validate again if the final release changes the tested code.
