import numpy as np
import pandas as pd
from obspy import UTCDateTime
from tqdm.auto import tqdm

from bats import get_statistics
from initial_conditions import observed_data

import matplotlib.pyplot as plt
from scipy.stats import probplot


def calculate_snr_and_variance(t, d, fs, ks):
    """Calculate only SNR and variance for one fixed signal model."""
    stats = get_statistics(
        t,
        d,
        fs,
        ks,
        calc_log_prob=False,
        calc_variance=True,
        calc_snr=True,
        calc_p_spec=False,
        calc_glob_ll=False,
        calc_cov_mat=False,
        calc_f_unc=False,
        calc_k_unc=False,
    )

    snr = float(stats.SNR)
    variance = float(stats.variance)

    if not np.isfinite(snr):
        raise FloatingPointError(
            f"Non-finite SNR returned for {len(fs)} signals"
        )

    if not np.isfinite(variance):
        raise FloatingPointError(
            f"Non-finite variance returned for {len(fs)} signals"
        )

    return snr, variance


def empirical_upper_tail_pvalue(observed, null_values):
    """One-sided Monte Carlo p-value for unusually large improvements.

    The +1 correction prevents a zero p-value from a finite number of
    Monte Carlo trials.
    """
    null_values = np.asarray(null_values, dtype=float)
    valid = np.isfinite(null_values)
    null_values = null_values[valid]

    if not np.isfinite(observed) or null_values.size == 0:
        return np.nan

    exceedance_count = np.count_nonzero(
        null_values >= observed
    )

    return float(
        (1 + exceedance_count)
        / (null_values.size + 1)
    )


def benjamini_hochberg(pvalues, alpha=0.05):
    """Apply Benjamini-Hochberg false-discovery-rate correction.

    Returns
    -------
    qvalues : numpy.ndarray
        Benjamini-Hochberg adjusted p-values.

    rejected : numpy.ndarray
        True where the null hypothesis is rejected at the requested FDR.
    """
    pvalues = np.asarray(pvalues, dtype=float)

    qvalues = np.full(pvalues.shape, np.nan, dtype=float)
    rejected = np.zeros(pvalues.shape, dtype=bool)

    valid_indices = np.flatnonzero(np.isfinite(pvalues))

    if valid_indices.size == 0:
        return qvalues, rejected

    valid_pvalues = pvalues[valid_indices]
    number_of_tests = valid_pvalues.size

    order = np.argsort(valid_pvalues)
    sorted_pvalues = valid_pvalues[order]

    ranks = np.arange(
        1,
        number_of_tests + 1,
        dtype=float,
    )

    sorted_qvalues = (
        sorted_pvalues
        * number_of_tests
        / ranks
    )

    # Enforce monotonicity of the adjusted p-values.
    sorted_qvalues = np.minimum.accumulate(
        sorted_qvalues[::-1]
    )[::-1]

    sorted_qvalues = np.clip(
        sorted_qvalues,
        0.0,
        1.0,
    )

    sorted_original_indices = valid_indices[order]
    qvalues[sorted_original_indices] = sorted_qvalues

    rejected[valid_indices] = (
        qvalues[valid_indices] <= alpha
    )

    return qvalues, rejected


def main():
    # ------------------------------------------------------------
    # Data configuration
    # ------------------------------------------------------------

    network = "IU"
    station = "KIP"
    location = "00"
    channel = "LHZ"
    stream_index = 0

    start_time = UTCDateTime("2025-07-29T23:24:50")
    end_time = UTCDateTime("2025-08-06T05:24:50")

    min_f = 0.0030
    max_f = 0.0040

    input_csv = (
        "C:/Users/starb/Downloads/"
        "dracula_output_3_4_KIP/N040_signals.csv"
    )

    output_csv = (
        "C:/Users/starb/Downloads/"
        "dracula_output_3_4_KIP/"
        "N040_signals_empirical_fdr.csv"
    )

    # ------------------------------------------------------------
    # Null-model configuration
    # ------------------------------------------------------------

    random_seed = 12345

    # With 32 tested signals, the smallest first-rank BH threshold is:
    #
    #     0.05 / 32 = 0.0015625
    #
    # A Monte Carlo p-value based on 1,000 trials has minimum possible
    # value 1 / 1001 = 0.000999, so it can resolve this threshold.
    n_noise_trials = 500

    fdr_alpha = 0.05

    # ------------------------------------------------------------
    # Load the time series
    # ------------------------------------------------------------

    t, d = observed_data(
        network=network,
        station=station,
        channel=channel,
        location=location,
        stream_index=stream_index,
        start_time=start_time,
        end_time=end_time,
        min_f=min_f,
        max_f=max_f,
    )

    t = np.asarray(t[200:10000], dtype=float)
    d = np.asarray(d[200:10000], dtype=float)

    if t.size != d.size:
        raise ValueError("t and d must have the same length")

    if t.size < 2:
        raise ValueError("The selected time series is too short")

    # ------------------------------------------------------------
    # Load the fitted frequencies and decay rates
    # ------------------------------------------------------------

    df = pd.read_csv(input_csv)

    required_columns = {
        "frequencies",
        "decay_rates",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Input CSV is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    fs = df["frequencies"].to_numpy(dtype=float)
    ks = df["decay_rates"].to_numpy(dtype=float)

    if fs.size != ks.size:
        raise ValueError(
            "The frequency and decay-rate columns must "
            "have the same length"
        )

    if fs.size < 2:
        raise ValueError(
            "At least two signals are required for "
            "leave-one-out testing"
        )

    if not np.all(np.isfinite(fs)):
        raise ValueError(
            "The frequency column contains non-finite values"
        )

    if not np.all(np.isfinite(ks)):
        raise ValueError(
            "The decay-rate column contains non-finite values"
        )

    n_signals = fs.size

    # ------------------------------------------------------------
    # Generate random stationary sinusoid frequencies
    # ------------------------------------------------------------

    rng = np.random.default_rng(random_seed)

    # The same random frequencies are used for every leave-one-out
    # model. This makes comparisons between rows less sensitive to
    # differences in their random draws.
    noise_fs = rng.uniform(
        low=min_f,
        high=max_f,
        size=n_noise_trials,
    )

    # The null component is a stationary sinusoid, not white noise.
    noise_k = 0.0

    # ------------------------------------------------------------
    # Calculate the complete-model statistics
    # ------------------------------------------------------------

    full_snr, full_variance = calculate_snr_and_variance(
        t,
        d,
        fs,
        ks,
    )

    print(f"Number of fitted signals: {n_signals}")
    print(f"Number of null trials per signal: {n_noise_trials}")
    print(f"Full-model SNR: {full_snr:.10g}")
    print(f"Full-model variance: {full_variance:.10g}")

    # ------------------------------------------------------------
    # Allocate result arrays
    # ------------------------------------------------------------

    # Observed improvement from adding each real signal back to its
    # corresponding leave-one-out model.
    snr_delta = np.empty(n_signals, dtype=float)
    variance_delta = np.empty(n_signals, dtype=float)

    snr_without_signal_values = np.empty(
        n_signals,
        dtype=float,
    )

    variance_without_signal_values = np.empty(
        n_signals,
        dtype=float,
    )

    # Null distributions produced by adding random stationary
    # sinusoids to each leave-one-out model.
    noise_snr_trials = np.empty(
        (n_signals, n_noise_trials),
        dtype=float,
    )

    noise_variance_trials = np.empty(
        (n_signals, n_noise_trials),
        dtype=float,
    )

    # One leave-one-out calculation plus all null calculations
    # for every signal.
    total_calculations = n_signals * (
        n_noise_trials + 1
    )

    pbar = tqdm(
        total=total_calculations,
        desc="Signal and random-frequency tests",
        unit="model",
    )

    try:
        for signal_index in range(n_signals):
            # ----------------------------------------------------
            # Remove the real signal associated with this CSV row
            # ----------------------------------------------------

            keep = np.ones(n_signals, dtype=bool)
            keep[signal_index] = False

            fs_without_signal = fs[keep]
            ks_without_signal = ks[keep]

            try:
                (
                    snr_without_signal,
                    variance_without_signal,
                ) = calculate_snr_and_variance(
                    t,
                    d,
                    fs_without_signal,
                    ks_without_signal,
                )
            except Exception as error:
                raise RuntimeError(
                    "Leave-one-out calculation failed for "
                    f"signal index {signal_index}, "
                    f"f={fs[signal_index]:.10g}, "
                    f"k={ks[signal_index]:.10g}"
                ) from error

            snr_without_signal_values[signal_index] = (
                snr_without_signal
            )

            variance_without_signal_values[signal_index] = (
                variance_without_signal
            )

            # SNR improvement from adding the real signal back.
            #
            # Positive means that the real signal increases SNR.
            snr_delta[signal_index] = (
                full_snr - snr_without_signal
            )

            # Variance improvement from adding the real signal back.
            #
            # This direction differs from the original script:
            #
            #     without signal - with signal
            #
            # Positive therefore means that adding the real signal
            # decreases residual variance.
            variance_delta[signal_index] = (
                variance_without_signal - full_variance
            )

            pbar.update(1)

            # ----------------------------------------------------
            # Random stationary-sinusoid null trials
            # ----------------------------------------------------

            for trial_index, noise_f in enumerate(noise_fs):
                # Add one random-frequency, non-decaying component
                # to the leave-one-out model. This gives the null
                # model the same number of components as the full
                # real-signal model.
                fs_with_noise = np.concatenate(
                    (
                        fs_without_signal,
                        np.asarray([noise_f], dtype=float),
                    )
                )

                ks_with_noise = np.concatenate(
                    (
                        ks_without_signal,
                        np.asarray([noise_k], dtype=float),
                    )
                )

                try:
                    (
                        snr_with_noise,
                        variance_with_noise,
                    ) = calculate_snr_and_variance(
                        t,
                        d,
                        fs_with_noise,
                        ks_with_noise,
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Random-frequency calculation failed for "
                        f"signal index {signal_index}, "
                        f"trial {trial_index}, "
                        f"noise frequency={noise_f:.10g}"
                    ) from error

                # Positive means that adding the random component
                # increased SNR.
                noise_snr_trials[
                    signal_index,
                    trial_index,
                ] = (
                    snr_with_noise
                    - snr_without_signal
                )

                # Positive means that adding the random component
                # reduced residual variance.
                noise_variance_trials[
                    signal_index,
                    trial_index,
                ] = (
                    variance_without_signal
                    - variance_with_noise
                )

                pbar.update(1)

    finally:
        pbar.close()

    # ------------------------------------------------------------
    # Summarize each signal's null distribution
    # ------------------------------------------------------------

    noise_snr_mean = np.mean(
        noise_snr_trials,
        axis=1,
    )

    noise_variance_mean = np.mean(
        noise_variance_trials,
        axis=1,
    )

    # ddof=1 gives the sample standard deviation.
    noise_snr_std = np.std(
        noise_snr_trials,
        axis=1,
        ddof=1,
    )

    noise_variance_std = np.std(
        noise_variance_trials,
        axis=1,
        ddof=1,
    )

    noise_snr_median = np.median(
        noise_snr_trials,
        axis=1,
    )

    noise_variance_median = np.median(
        noise_variance_trials,
        axis=1,
    )

    noise_snr_95th_percentile = np.quantile(
        noise_snr_trials,
        0.95,
        axis=1,
    )

    noise_variance_95th_percentile = np.quantile(
        noise_variance_trials,
        0.95,
        axis=1,
    )

    # ------------------------------------------------------------
    # Standardized differences from the null means
    # ------------------------------------------------------------

    snr_delta_zscore = np.divide(
        snr_delta - noise_snr_mean,
        noise_snr_std,
        out=np.full(n_signals, np.nan, dtype=float),
        where=noise_snr_std > 0.0,
    )

    variance_delta_zscore = np.divide(
        variance_delta - noise_variance_mean,
        noise_variance_std,
        out=np.full(n_signals, np.nan, dtype=float),
        where=noise_variance_std > 0.0,
    )

    # ------------------------------------------------------------
    # One-sided empirical Monte Carlo p-values
    # ------------------------------------------------------------
    #
    # The alternative hypothesis is directional:
    #
    #     real signal improvement > random-signal improvement
    #
    # The +1 correction prevents zero p-values.

    snr_empirical_pvalue = np.empty(
        n_signals,
        dtype=float,
    )

    variance_empirical_pvalue = np.empty(
        n_signals,
        dtype=float,
    )

    for signal_index in range(n_signals):
        snr_empirical_pvalue[signal_index] = (
            empirical_upper_tail_pvalue(
                observed=snr_delta[signal_index],
                null_values=noise_snr_trials[signal_index],
            )
        )

        variance_empirical_pvalue[signal_index] = (
            empirical_upper_tail_pvalue(
                observed=variance_delta[signal_index],
                null_values=(
                    noise_variance_trials[signal_index]
                ),
            )
        )

    # Descriptive noise-rejection scores. These are not posterior
    # probabilities that the fitted signals are physically real.
    snr_noise_rejection_percent = (
        100.0 * (1.0 - snr_empirical_pvalue)
    )

    variance_noise_rejection_percent = (
        100.0 * (1.0 - variance_empirical_pvalue)
    )

    # ------------------------------------------------------------
    # Benjamini-Hochberg FDR correction
    # ------------------------------------------------------------
    #
    # SNR and variance are corrected separately because they are
    # related diagnostics, not independent pieces of evidence.

    (
        snr_bh_qvalue,
        snr_bh_rejected,
    ) = benjamini_hochberg(
        snr_empirical_pvalue,
        alpha=fdr_alpha,
    )

    (
        variance_bh_qvalue,
        variance_bh_rejected,
    ) = benjamini_hochberg(
        variance_empirical_pvalue,
        alpha=fdr_alpha,
    )

    # ------------------------------------------------------------
    # Add results to the original signal table
    # ------------------------------------------------------------

    # One-based index is easier to compare with displayed CSV rows.
    df["confidence_signal_index"] = np.arange(
        1,
        n_signals + 1,
    )

    # Complete-model and leave-one-out values.
    df["full_model_snr"] = full_snr
    df["snr_without_signal"] = (
        snr_without_signal_values
    )

    df["full_model_variance"] = full_variance
    df["variance_without_signal"] = (
        variance_without_signal_values
    )

    # Observed real-signal improvements.
    #
    # Positive SNR delta:
    #     adding the real signal increased SNR.
    #
    # Positive variance delta:
    #     adding the real signal reduced residual variance.
    df["snr_delta"] = snr_delta
    df["variance_delta"] = variance_delta

    # Random stationary-sinusoid null summaries.
    df["noise_snr_delta_mean"] = noise_snr_mean
    df["noise_snr_delta_median"] = noise_snr_median
    df["noise_snr_delta_std"] = noise_snr_std
    df["noise_snr_delta_95th_percentile"] = (
        noise_snr_95th_percentile
    )

    df["noise_variance_delta_mean"] = (
        noise_variance_mean
    )
    df["noise_variance_delta_median"] = (
        noise_variance_median
    )
    df["noise_variance_delta_std"] = (
        noise_variance_std
    )
    df["noise_variance_delta_95th_percentile"] = (
        noise_variance_95th_percentile
    )

    # Number of null standard deviations between the observed
    # improvement and the mean null improvement.
    df["snr_delta_zscore"] = snr_delta_zscore
    df["variance_delta_zscore"] = (
        variance_delta_zscore
    )

    # Uncorrected one-sided empirical p-values.
    df["snr_empirical_pvalue"] = (
        snr_empirical_pvalue
    )
    df["variance_empirical_pvalue"] = (
        variance_empirical_pvalue
    )

    # Descriptive versions of 100 * (1 - p).
    df["snr_noise_rejection_percent"] = (
        snr_noise_rejection_percent
    )
    df["variance_noise_rejection_percent"] = (
        variance_noise_rejection_percent
    )

    # Benjamini-Hochberg adjusted p-values and decisions.
    df["snr_bh_qvalue"] = snr_bh_qvalue
    df["snr_bh_reject_fdr_0_05"] = (
        snr_bh_rejected
    )

    df["variance_bh_qvalue"] = (
        variance_bh_qvalue
    )
    df["variance_bh_reject_fdr_0_05"] = (
        variance_bh_rejected
    )

    # Helpful direct comparisons with the upper 5% null threshold.
    df["snr_above_noise_95th_percentile"] = (
        snr_delta > noise_snr_95th_percentile
    )

    df["variance_above_noise_95th_percentile"] = (
        variance_delta
        > noise_variance_95th_percentile
    )

    snr_percentile = np.empty(n_signals, dtype=float)
    variance_percentile = np.empty(n_signals, dtype=float)

    for signal_index in range(n_signals):
        snr_percentile[signal_index] = (
            100.0
            * np.mean(
                noise_snr_trials[signal_index]
                < snr_delta[signal_index]
            )
        )

        variance_percentile[signal_index] = (
            100.0
            * np.mean(
                noise_variance_trials[signal_index]
                < variance_delta[signal_index]
            )
        )

    df["snr_null_percentile"] = snr_percentile
    df["variance_null_percentile"] = variance_percentile

    for signal_index in range(n_signals):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        null_values = noise_snr_trials[signal_index]

        axes[0].hist(
            null_values,
            bins=40,
            density=True,
            color="0.7",
            edgecolor="black",
        )
        axes[0].axvline(
            snr_delta[signal_index],
            color="red",
            linewidth=2,
            label="Real signal",
        )
        axes[0].set_title(
            f"Signal {signal_index + 1} SNR null"
        )
        axes[0].legend()

        probplot(
            null_values,
            dist="norm",
            plot=axes[1],
        )
        axes[1].set_title("Normal Q-Q plot")

        fig.tight_layout()
        fig.savefig(
            f"signal_{signal_index + 1:03d}_snr_null.png",
            dpi=150,
        )
        plt.close(fig)

    # ------------------------------------------------------------
    # Save the expanded signal table
    # ------------------------------------------------------------

    df.to_csv(
        output_csv,
        index=False,
    )

    # ------------------------------------------------------------
    # Print a short summary
    # ------------------------------------------------------------

    number_snr_rejected = int(
        np.count_nonzero(snr_bh_rejected)
    )

    number_variance_rejected = int(
        np.count_nonzero(variance_bh_rejected)
    )

    number_both_rejected = int(
        np.count_nonzero(
            snr_bh_rejected
            & variance_bh_rejected
        )
    )

    print()
    print(f"Saved confidence statistics to: {output_csv}")
    print(
        "SNR detections at BH FDR 0.05: "
        f"{number_snr_rejected}/{n_signals}"
    )
    print(
        "Variance detections at BH FDR 0.05: "
        f"{number_variance_rejected}/{n_signals}"
    )
    print(
        "Signals passing both diagnostics: "
        f"{number_both_rejected}/{n_signals}"
    )

    # Display signals ordered by their SNR-adjusted p-values.
    summary_columns = [
        "confidence_signal_index",
        "frequencies",
        "decay_rates",
        "snr_delta",
        "noise_snr_delta_mean",
        "snr_delta_zscore",
        "snr_empirical_pvalue",
        "snr_bh_qvalue",
        "snr_bh_reject_fdr_0_05",
        "variance_delta",
        "noise_variance_delta_mean",
        "variance_delta_zscore",
        "variance_empirical_pvalue",
        "variance_bh_qvalue",
        "variance_bh_reject_fdr_0_05",
    ]

    summary = df[summary_columns].sort_values(
        "snr_bh_qvalue",
        ascending=True,
    )

    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
