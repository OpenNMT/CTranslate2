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

CONFIRMATION_MIN_PAIRS = 5
CONFIRMATION_MAX_PAIRS = 9
MAX_RELATIVE_IQR = 0.10


def relative_iqr(values):
    """Estimate timing spread using the middle half of the samples."""
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    median = statistics.median(values)
    return (q3 - q1) / abs(median) if median else (0.0 if q3 == q1 else math.inf)

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


def run_task(py, task, limit, device, model_dir=None, command="run", compute_type="default"):
    cmd = [py, str(ROOT / "tasks.py"), command, task, "--device", device]
    if command == "run":
        cmd += ["--limit", str(limit if limit is not None else 0), "--compute-type", compute_type]
    if model_dir is not None:
        cmd += ["--model-dir", model_dir]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")
    out = subprocess.check_output(cmd, env=env).decode()
    return json.loads([l for l in out.splitlines() if l.strip()][-1])


def select_compute_types(base_py, cand_py, task, device, model_dirs):
    resolved = {}
    for label, py in (("baseline", base_py), ("candidate", cand_py)):
        resolved[label] = run_task(py, task, None, device,
                                   model_dirs[label].get(task), command="resolve")
        for requested in ("default", "auto"):
            effective = resolved[label].get(requested)
            if not isinstance(effective, str) or not effective or effective in ("default", "auto"):
                raise ValueError(f"{task}: {label} invalid resolved {requested} type: {effective!r}")
            print(f"[gate] {task}: {label} requested={requested} effective={effective}", flush=True)
    if all(types["auto"] == types["default"] for types in resolved.values()):
        print(f"[gate] {task}: skipping auto; it matches default on both wheels", flush=True)
        return ("default",)
    return ("default", "auto")


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
            or any(not isinstance(s, str) for s in baseline + candidate)):
        raise ValueError("Expected equal, nonempty sets of string completions")
    changed = [(i, b, c) for i, (b, c) in enumerate(zip(baseline, candidate), 1) if b != c]
    print(f"[gate] changed completions: {len(changed)}/{len(baseline)} [informational]")
    for i, b, c in changed:
        print(f"  prompt {i}: baseline={b!r} candidate={c!r}")
    return len(changed)


def measure_round(base_py, cand_py, task, limit, device, model_dirs, confirmation=False, compute_type="default"):
    """Compare alternating pairs, adapting confirmation to measurement spread."""
    samples = {"baseline": [], "candidate": []}
    quality_failed = False
    max_pairs = CONFIRMATION_MAX_PAIRS if confirmation else 3
    min_pairs = CONFIRMATION_MIN_PAIRS if confirmation else 3
    performance_metrics = [metric for metric, rule in RULES[task].items() if rule[2]]
    stable = False
    for repeat in range(max_pairs):
        order = ("baseline", "candidate")
        if (repeat + int(confirmation)) % 2:
            order = tuple(reversed(order))
        for label in order:
            print(f"[gate] {task}: {label} measurement {repeat + 1}/{max_pairs}", flush=True)
            result = run_task(base_py if label == "baseline" else cand_py,
                              task, limit, device,
                              model_dir=model_dirs[label].get(task), compute_type=compute_type)
            validate_metrics(task, result, label)
            samples[label].append(result)
        base, cand = samples["baseline"][-1], samples["candidate"][-1]
        if task == "t5gemma_generation":
            compare_completions(base, cand)
        quality_failed |= any(result.failed for result in judge(task, base, cand).values()
                              if not result.performance)

        if repeat + 1 >= min_pairs:
            spreads = {
                (label, metric): relative_iqr([result[metric] for result in results])
                for label, results in samples.items() for metric in performance_metrics
            }
            stable = all(spread <= MAX_RELATIVE_IQR for spread in spreads.values())
            for (label, metric), spread in spreads.items():
                print(f"[gate] {task}: {label}.{metric} IQR/median={spread:.1%} "
                      f"({repeat + 1} pairs)", flush=True)
            if not confirmation or stable:
                break

    if confirmation and not stable:
        print(f"[gate] {task}: confirmation measurement budget exhausted; "
              "timings remain noisy, applying median thresholds", flush=True)

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


def evaluate_task(base_py, cand_py, task, limit, device, model_dirs, compute_type="default"):
    quality_failed, performance_failures = measure_round(
        base_py, cand_py, task, limit, device, model_dirs, compute_type=compute_type)
    if performance_failures:
        print(f"[gate] {task}: confirming performance regression", flush=True)
        confirmation_quality_failed, confirmed = measure_round(
            base_py, cand_py, task, limit, device, model_dirs, confirmation=True, compute_type=compute_type)
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
            for compute_type in select_compute_types(base_py, cand_py, task, args.device, model_dirs):
                print(f"[gate] {task}: compute_type={compute_type}", flush=True)
                failed |= evaluate_task(base_py, cand_py, task, task_limit,
                                        args.device, model_dirs, compute_type=compute_type)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
