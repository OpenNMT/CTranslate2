# Release validation

1. Open a release PR and wait for CI to pass.
2. On each Linux x86-64 GPU machine, check out the PR head and build its native
   tests locally. Install test dependencies, model data, GPU drivers, and runtime
   libraries, including a GPU-compatible PyTorch. Authenticate with `gh auth login`
   using an account permitted to download artifacts and write commit statuses.
3. Run the native and Python tests, including GPU tests:

   ```bash
   bash tools/release-check.sh cuda
   ```

   Replace `cuda` with `rocm` on the AMD machine.

The script finds the current branch's PR in the GitHub repository named by `origin`.
It checks that tracked files are clean and HEAD matches the PR, then
publishes `release/cuda` or `release/rocm` as pending, then success or failure.
It downloads the CUDA or ROCm wheel from the latest push CI run for that exact
commit. Wait for the wheel artifact to be uploaded before running validation.
The wheel is installed into a temporary environment that reuses your Python
dependencies; the environment is removed afterwards. Your installed CTranslate2
is left untouched. Native tests use your local build, since CI does not upload it.
The script requires a visible GPU and both test commands to pass.
Detailed output stays in the terminal.

Configure both statuses as required checks for release PRs. Re-run validation
for every new commit. Publish the validated artifacts after CI and both GPU
checks pass; validate again if the final release changes the tested code.
