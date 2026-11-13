"""Release gate: compare candidate ctranslate2 against latest PyPI release.

Creates two venvs, runs the same tasks in each, exits non-zero on quality
regressions or confirmed performance regressions.
"""

import argparse
from collections import namedtuple
import glob
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import urllib.request
import venv

ROOT = pathlib.Path(__file__).resolve().parent

# (threshold, higher_is_better, relative).
# Relative thresholds are fractions of the baseline; others are absolute deltas.
RULES = {
    "whisper_wer": {
        "wer":         (0.01,  False, False),   # +1 WER percentage point
        "rtf_inverse": (0.10,  True,  True),   # >10% throughput drop
    },
    "opus_bleu": {
        "bleu":           (1.0,  True, False),  # -1 BLEU point
        "tokens_per_sec": (0.10, True, True),
    },
    "t5gemma_generation": {
        "tokens_per_sec": (0.10, True, True),
    },
}


MetricResult = namedtuple("MetricResult", "baseline candidate failed performance")

def latest_version(pkg="ctranslate2"):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
        return json.load(r)["info"]["version"]


def make_venv(path, ct2_spec):
    venv.EnvBuilder(with_pip=True, clear=True).create(path)
    if sys.platform == "win32":
        py = os.path.join(path, "Scripts", "python.exe")
    else:
        py = os.path.join(path, "bin", "python")
    subprocess.check_call([py, "-m", "pip", "install", "-q", "-U", "pip"])
    subprocess.check_call([py, "-m", "pip", "install", "-q", ct2_spec,
                           "-r", str(ROOT / "requirements.txt")])
    return py


CONVERTIBLE_TASKS = ("opus_bleu", "t5gemma_generation")


def convert_models(py, out_root):
    """Convert each CT2 model once per venv; return {task: path}."""
    dirs = {}
    for task in CONVERTIBLE_TASKS:
        out_dir = os.path.join(out_root, task)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[gate] converting {task} with {py}", flush=True)
        subprocess.check_call([py, str(ROOT / "tasks.py"), "convert", task, out_dir])
        dirs[task] = out_dir
    return dirs


def run_task(py, task, limit, device, model_dir=None):
    cmd = [py, str(ROOT / "tasks.py"), "run", task, "--device", device]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if model_dir is not None:
        cmd += ["--model-dir", model_dir]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")
    out = subprocess.check_output(cmd, env=env).decode()
    return json.loads([l for l in out.splitlines() if l.strip()][-1])


def validate_metrics(task, result, label):
    for metric in RULES[task]:
        value = result.get(metric)
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(
                f"{task}: {label} metric {metric!r} "
                f"is missing or invalid: {value!r}")


def judge(task, base, cand):
    validate_metrics(task, base, "baseline")
    validate_metrics(task, cand, "candidate")
    results = {}
    for metric, (threshold, higher_is_better, relative) in RULES[task].items():
        b, c = base[metric], cand[metric]
        regression = (b - c) if higher_is_better else (c - b)
        limit = threshold * abs(b) if relative else threshold
        results[metric] = MetricResult(b, c, regression > limit, relative)
    return results


def compare_completions(base, cand):
    baseline, candidate = base["outputs"], cand["outputs"]
    if (not baseline or len(baseline) != len(candidate)
            or any(not isinstance(s, str) or not s.strip() for s in baseline + candidate)):
        raise ValueError("Expected equal, nonempty sets of nonempty completions")
    changed = [(i, b, c) for i, (b, c) in enumerate(zip(baseline, candidate), 1) if b != c]
    print(f"[gate] changed completions: {len(changed)}/{len(baseline)} [informational]")
    for i, b, c in changed:
        print(f"  prompt {i}: baseline={b!r} candidate={c!r}")
    return len(changed)


def measure_round(base_py, cand_py, task, limit, device, model_dirs, confirmation=False):
    """Compare three sequential pairs; retain every quality failure."""
    samples = {"baseline": [], "candidate": []}
    quality_failed = False
    for repeat in range(3):
        order = ("baseline", "candidate")
        if (repeat + int(confirmation)) % 2:
            order = tuple(reversed(order))
        for label in order:
            print(f"[gate] {task}: {label} measurement {repeat + 1}/3", flush=True)
            result = run_task(base_py if label == "baseline" else cand_py,
                              task, limit, device,
                              model_dir=model_dirs[label].get(task))
            validate_metrics(task, result, label)
            samples[label].append(result)
        base, cand = samples["baseline"][-1], samples["candidate"][-1]
        if task == "t5gemma_generation":
            compare_completions(base, cand)
        quality_failed |= any(result.failed for result in judge(task, base, cand).values()
                              if not result.performance)

    medians = {}
    for label, results in samples.items():
        medians[label] = {
            metric: statistics.median(result[metric] for result in results)
            for metric in RULES[task]
        }
        print(f"[gate] {task}: {label} measurements={results}")
    results = judge(task, medians["baseline"], medians["candidate"])
    for metric, result in results.items():
        print(f"  {task}.{metric} median: {result.baseline:.4f} -> {result.candidate:.4f} "
              f"({result.candidate - result.baseline:+.4f}) {'FAIL' if result.failed else 'ok'}")
    performance_failures = {metric for metric, result in results.items()
                            if result.performance and result.failed}
    return quality_failed, performance_failures


def evaluate_task(base_py, cand_py, task, limit, device, model_dirs):
    quality_failed, performance_failures = measure_round(
        base_py, cand_py, task, limit, device, model_dirs)
    if performance_failures:
        print(f"[gate] {task}: confirming performance regression", flush=True)
        confirmation_quality_failed, confirmed = measure_round(
            base_py, cand_py, task, limit, device, model_dirs, confirmation=True)
        quality_failed |= confirmation_quality_failed
        performance_failures &= confirmed
        print(f"[gate] {task}: performance regression "
              f"{'confirmed' if performance_failures else 'not confirmed'}")
    return quality_failed or bool(performance_failures)


def main():
    p = argparse.ArgumentParser()
    candidate = p.add_mutually_exclusive_group()
    candidate.add_argument("--candidate-wheel", default="ctranslate2",
                   help="pip spec or wheel path (defaults to PyPI = self-test)")
    candidate.add_argument("--candidate-wheel-pattern",
                           help="glob matching exactly one candidate wheel")
    p.add_argument("--limit", type=int, default=50, help="0 = full task set")
    p.add_argument("--bleu-limit", type=int, default=500,
                   help="number of translation sentences (default: 500; overrides --limit)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.bleu_limit < 1:
        p.error("--bleu-limit must be positive")
    if args.candidate_wheel_pattern:
        wheels = glob.glob(args.candidate_wheel_pattern)
        if len(wheels) != 1:
            p.error(f"Expected exactly one candidate wheel, found {len(wheels)}: {wheels}")
        args.candidate_wheel = str(pathlib.Path(wheels[0]).resolve())
    limit = None if args.limit == 0 else args.limit

    baseline_spec = f"ctranslate2=={latest_version()}"
    print(f"[gate] baseline={baseline_spec} candidate={args.candidate_wheel}")

    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        print("[gate] creating baseline venv")
        base_py = make_venv(os.path.join(tmp, "baseline"), baseline_spec)
        print("[gate] creating candidate venv")
        cand_py = make_venv(os.path.join(tmp, "candidate"), args.candidate_wheel)

        model_dirs = {
            "baseline": convert_models(base_py, os.path.join(tmp, "baseline-models")),
            "candidate": convert_models(cand_py, os.path.join(tmp, "candidate-models")),
        }

        failed = False
        for task in RULES:
            task_limit = args.bleu_limit if task == "opus_bleu" else limit
            if task == "t5gemma_generation":
                task_limit = 50
            failed |= evaluate_task(base_py, cand_py, task, task_limit,
                                    args.device, model_dirs)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
