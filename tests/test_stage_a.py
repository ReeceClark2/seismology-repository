"""Stage A: constructor, argument routing, output layout, progress mode."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from colony import Colony, resolve_progress_mode, resolve_signals_per_worker
from dracula import Dracula, create_run_layout, resolve_signal_counts


def _series(n: int = 32) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float)
    d = np.sin(2.0 * np.pi * 0.05 * t)
    return t, d


class ConstructorTests(unittest.TestCase):
    def test_constructs_with_initial_conditions(self) -> None:
        t, d = _series()
        model = Dracula(t, d, f_init=[0.05], k_init=[0.01])
        self.assertEqual(int(model.f_init.shape[0]), 1)
        self.assertEqual(int(model.k_init.shape[0]), 1)

    def test_constructs_without_initial_conditions(self) -> None:
        t, d = _series()
        model = Dracula(t, d)
        self.assertIsNone(model.f_init)
        self.assertIsNone(model.k_init)

    def test_rejects_only_one_initial_array(self) -> None:
        t, d = _series()
        with self.assertRaises(ValueError):
            Dracula(t, d, f_init=[0.05])
        with self.assertRaises(ValueError):
            Dracula(t, d, k_init=[0.01])

    def test_rejects_nonpositive_decay(self) -> None:
        t, d = _series()
        with self.assertRaises(ValueError):
            Dracula(t, d, f_init=[0.05], k_init=[0.0])
        with self.assertRaises(ValueError):
            Dracula(t, d, f_init=[0.05], k_init=[-0.01])

    def test_rejects_mismatched_lengths_and_nonincreasing_time(self) -> None:
        t, d = _series()
        with self.assertRaises(ValueError):
            Dracula(t, d[:-1])
        with self.assertRaises(ValueError):
            Dracula(t[::-1], d)
        with self.assertRaises(ValueError):
            Dracula(t, d, f_init=[0.05, 0.06], k_init=[0.01])

    def test_dispatch_without_initial_conditions_fails_before_sampling(self) -> None:
        t, d = _series()
        model = Dracula(t, d)
        with self.assertRaises(ValueError):
            model.dispatch(signals_per_worker=2, signals=1)


class SignalCountTests(unittest.TestCase):
    def test_signals_sets_exact_range(self) -> None:
        self.assertEqual(resolve_signal_counts(32, None, None), (32, 32))

    def test_min_max_range(self) -> None:
        self.assertEqual(resolve_signal_counts(None, 20, 40), (20, 40))

    def test_rejects_signals_with_min_or_max(self) -> None:
        with self.assertRaises(ValueError):
            resolve_signal_counts(32, 20, None)
        with self.assertRaises(ValueError):
            resolve_signal_counts(32, None, 40)
        with self.assertRaises(ValueError):
            resolve_signal_counts(32, 20, 40)

    def test_requires_both_min_and_max_or_signals(self) -> None:
        with self.assertRaises(ValueError):
            resolve_signal_counts(None, None, None)
        with self.assertRaises(ValueError):
            resolve_signal_counts(None, 20, None)
        with self.assertRaises(ValueError):
            resolve_signal_counts(None, None, 40)


class WorkerRenameTests(unittest.TestCase):
    def test_signals_per_worker_required(self) -> None:
        with self.assertRaises(TypeError):
            resolve_signals_per_worker()

    def test_rejects_both_aliases(self) -> None:
        with self.assertRaises(ValueError):
            resolve_signals_per_worker(signals_per_worker=4, f_per_worker=4)

    def test_deprecated_f_per_worker_alias(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = resolve_signals_per_worker(f_per_worker=7)
        self.assertEqual(value, 7)
        self.assertTrue(
            any(issubclass(item.category, DeprecationWarning) for item in caught)
        )

    def test_colony_chunks_with_signals_per_worker(self) -> None:
        t, d = _series()
        colony = Colony(t, d, f_init=[0.04, 0.05, 0.06], k_init=[0.01, 0.02, 0.03])
        tasks = colony.get_tasks(signals_per_worker=2, progress_mode="main")
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].progress_mode, "main")
        self.assertEqual(int(np.asarray(tasks[0].f_init).size), 2)
        self.assertEqual(int(np.asarray(tasks[1].f_init).size), 1)


class ProgressModeTests(unittest.TestCase):
    def test_valid_modes(self) -> None:
        self.assertEqual(resolve_progress_mode("none"), "none")
        self.assertEqual(resolve_progress_mode("main"), "main")
        self.assertEqual(resolve_progress_mode("detailed"), "detailed")

    def test_invalid_mode(self) -> None:
        with self.assertRaises(ValueError):
            resolve_progress_mode("quiet")


class OutputLayoutTests(unittest.TestCase):
    def test_creates_stage_directories_and_config_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = create_run_layout(Path(tmp) / "dracula_output")
            self.assertTrue(layout.root.exists())
            self.assertTrue(layout.run_configuration.is_dir())
            self.assertTrue(layout.initial_conditions.is_dir())
            self.assertTrue(layout.summary.is_dir())
            self.assertTrue(layout.sampling.is_dir())
            self.assertTrue(layout.final_results.is_dir())
            self.assertTrue(layout.root.name.startswith("dracula_output_"))


class DispatchValidationTests(unittest.TestCase):
    def test_signals_conflict_is_rejected_on_dispatch(self) -> None:
        t, d = _series()
        model = Dracula(t, d, f_init=[0.04, 0.05], k_init=[0.01, 0.02])
        with self.assertRaises(ValueError):
            model.dispatch(
                signals_per_worker=1,
                signals=1,
                min_signals=1,
                max_signals=2,
            )

    def test_progress_mode_rejected_before_sampling(self) -> None:
        t, d = _series()
        model = Dracula(t, d, f_init=[0.04], k_init=[0.01])
        with self.assertRaises(ValueError):
            model.dispatch(
                signals_per_worker=1,
                signals=1,
                progress_mode="silent",
            )


if __name__ == "__main__":
    unittest.main()
