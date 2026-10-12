# Release validation

1. Open a release PR and wait for CI to pass.
2. On each GPU machine, check out the PR head and build/install its candidate
   wheel and native tests. Authenticate with `gh auth login` using an account
   permitted to write commit statuses.
3. Run your validation command, including GPU tests, `../ct2-eval`, and benchmarks:

   ```bash
   bash release-check.sh cuda PR_URL COMMAND [ARG...]
   bash release-check.sh rocm PR_URL COMMAND [ARG...]
   ```

The script checks that tracked files are clean and HEAD matches the PR. It
publishes `release/cuda` or `release/rocm` as pending, then success or failure.
The backend argument only labels the status; configure GPU execution yourself.
Your command must fail on regressions or missing GPU coverage. Compare against
the previous release on the same hardware. Detailed output stays in the terminal.

Configure both statuses as required checks for release PRs. Re-run validation
for every new commit. Publish the validated artifacts after CI and both GPU
checks pass; validate again if the final release changes the tested code.
