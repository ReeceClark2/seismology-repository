"""Stage B: natural-log decay parameterization and imposed sampling surfaces."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from bats import (
    BATS,
    get_statistics,
    log_k_to_physical,
    multiplicative_decay_interval,
    physical_to_log_k,
    resolve_bounds_mode,
    resolve_imposed_surface,
)
from dracula import Dracula


def _series(n: int = 256) -> tuple[np.ndarray, np.ndarray, float, float]:
    t = np.linspace(0.0, 200.0, n)
    frequency = 0.04
    decay = 0.012
    data = np.exp(-decay * t) * np.cos(2.0 * np.pi * frequency * t)
    return t, data, frequency, decay


class LogKTransformTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        k = np.array([1e-6, 0.01, 2.5])
        log_k = physical_to_log_k(k)
        recovered = np.asarray(log_k_to_physical(log_k))
        np.testing.assert_allclose(recovered, k, rtol=1e-12)

    def test_rejects_nonpositive_decay(self) -> None:
        with self.assertRaises(ValueError):
            physical_to_log_k([0.0])
        with self.assertRaises(ValueError):
            physical_to_log_k([-1e-3])

    def test_multiplicative_interval(self) -> None:
        k = np.array([0.02, 0.05])
        sigma = np.array([0.1, 0.2])
        factor, lower, upper = multiplicative_decay_interval(k, sigma)
        np.testing.assert_allclose(factor, np.exp(sigma))
        np.testing.assert_allclose(lower, k / np.exp(sigma))
        np.testing.assert_allclose(upper, k * np.exp(sigma))
        self.assertTrue(np.all(np.asarray(lower) < k))
        self.assertTrue(np.all(k < np.asarray(upper)))


class ImposedSurfaceTests(unittest.TestCase):
    def test_default_gaussian(self) -> None:
        self.assertEqual(resolve_imposed_surface(), "gaussian")

    def test_deprecated_unbounded_mapping(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(resolve_imposed_surface(unbounded=True), "uniform")
            self.assertEqual(resolve_imposed_surface(unbounded=False), "gaussian")
        self.assertTrue(
            any(issubclass(item.category, DeprecationWarning) for item in caught)
        )

    def test_rejects_conflicting_aliases(self) -> None:
        with self.assertRaises(ValueError):
            resolve_imposed_surface(imposed_surface="gaussian", unbounded=True)

    def test_rejects_unknown_surface(self) -> None:
        with self.assertRaises(ValueError):
            resolve_imposed_surface(imposed_surface="cauchy")


class BoundsModeTests(unittest.TestCase):
    def test_bandwidth_mode(self) -> None:
        self.assertEqual(resolve_bounds_mode("bandwidth"), "bandwidth")

    def test_subband_mode_is_available(self) -> None:
        self.assertEqual(resolve_bounds_mode("initial_subband"), "initial_subband")


class StatisticsLogKTests(unittest.TestCase):
    def test_covariance_is_frequency_then_log_decay(self) -> None:
        t, data, frequency, decay = _series()
        stats = get_statistics(
            t,
            data,
            [frequency, frequency * 1.3],
            [decay, decay * 1.2],
            calc_p_spec=False,
            calc_glob_ll=False,
        )
        self.assertEqual(tuple(stats.extras["parameter_order"]), ("f", "log_k"))
        self.assertEqual(np.asarray(stats.cov_mat).shape, (4, 4))
        np.testing.assert_allclose(
            np.asarray(stats.log_ks),
            np.log([decay, decay * 1.2]),
            rtol=1e-12,
        )
        sigma = np.asarray(stats.sigma_log_k)
        ks = np.asarray(stats.ks)
        self.assertTrue(np.all(sigma > 0.0))
        np.testing.assert_allclose(
            np.asarray(stats.k_unc),
            ks * sigma,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(stats.k_lower),
            ks / np.exp(sigma),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(stats.k_upper),
            ks * np.exp(sigma),
            rtol=1e-12,
        )


class GridSearchLogKTests(unittest.TestCase):
    def test_rejects_nonpositive_min_k(self) -> None:
        t, data, frequency, decay = _series()
        model = BATS(t, data, [frequency], [decay])
        with self.assertRaises(ValueError):
            model.run_grid_search(
                min_f=0.02,
                max_f=0.06,
                min_k=0.0,
                max_k=0.05,
                f_points=4,
                k_points=4,
                progress_bar=False,
            )


class DispatchSurfaceTests(unittest.TestCase):
    def test_conflicting_surface_rejected_before_sampling(self) -> None:
        t, data, frequency, decay = _series()
        model = Dracula(t, data, f_init=[frequency], k_init=[decay])
        with self.assertRaises(ValueError):
            model.dispatch(
                signals_per_worker=1,
                signals=1,
                imposed_surface="gaussian",
                unbounded=True,
            )

    def test_subband_bounds_require_candidates(self) -> None:
        t, data, frequency, decay = _series()
        model = Dracula(t, data, f_init=[frequency], k_init=[decay])
        with self.assertRaises(ValueError):
            model.dispatch(
                signals_per_worker=1,
                signals=1,
                bounds_mode="initial_subband",
            )


if __name__ == "__main__":
    unittest.main()
