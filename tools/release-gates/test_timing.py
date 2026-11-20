"""Check warm-up consumption and timing boundaries without model dependencies."""

import unittest
from unittest.mock import patch

from tasks import run_and_measure


class TimingTest(unittest.TestCase):
    def test_warmup_is_consumed_before_timing(self):
        events = []

        def infer(batch, scale):
            for value in batch:
                events.append(value)
                yield value * scale

        def clock():
            events.append("clock")
            return len(events)

        with patch("tasks.time.perf_counter", side_effect=clock):
            results, elapsed = run_and_measure(infer, [1, 2, 3, 4], warmup_size=2, scale=10)
        self.assertEqual(events, [1, 2, "clock", 1, 2, 3, 4, "clock"])
        self.assertEqual(results, [10, 20, 30, 40])
        self.assertEqual(elapsed, 5)

    def test_warmup_is_capped_so_measured_pass_sees_fresh_inputs(self):
        seen = []

        def infer(batch):
            for value in batch:
                seen.append(value)
                yield value

        with patch("tasks.time.perf_counter", side_effect=[0.0, 1.0]):
            run_and_measure(infer, [1, 2, 3], warmup_size=8)
        self.assertEqual(seen, [1, 1, 2, 3])
