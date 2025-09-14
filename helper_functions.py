import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from DraggableLine import DraggableHorizontalLine, DraggableVerticalLine
from matplotlib.collections import PolyCollection
try:
    from sklearn.mixture import GaussianMixture
except Exception:
    GaussianMixture = None
from matplotlib.widgets import SpanSelector
def detect_time_signal_columns(df: pd.DataFrame):
    candidates_time = ["time", "t", "timestamp", "sec", "seconds", "x"]
    candidates_signal = ["signal", "intensity", "value", "amplitude", "y"]
    lower_cols = {c.lower(): c for c in df.columns}

    time_col = None
    signal_col = None
    for name in candidates_time:
        if name in lower_cols:
            time_col = lower_cols[name]
            break
    for name in candidates_signal:
        if name in lower_cols and lower_cols[name] != time_col:
            signal_col = lower_cols[name]
            break

    if time_col is None or signal_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            best_time = None
            for c in numeric_cols:
                series = df[c].dropna()
                if series.is_monotonic_increasing:
                    best_time = c
                    break
            time_col = time_col or best_time or numeric_cols[0]
            remaining = [c for c in numeric_cols if c != time_col]
            signal_col = signal_col or (remaining[0] if remaining else numeric_cols[0])
        elif len(df.columns) >= 2:
            time_col = df.columns[0]
            signal_col = df.columns[1]
        else:
            raise ValueError("Unable to detect suitable Time and Signal columns.")

    return time_col, signal_col

def detect_time_label_columns(df: pd.DataFrame):
    lower_cols = {c.lower(): c for c in df.columns}
    # Prefer explicit names
    time_col = lower_cols.get('time')
    label_col = lower_cols.get('label')
    if time_col and label_col:
        return time_col, label_col
    # Fallback heuristics
    # Time-like: numeric, monotonic increasing
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    best_time = None
    for c in numeric_cols:
        series = df[c].dropna()
        if series.is_monotonic_increasing:
            best_time = c
            break
    time_col = time_col or best_time or (numeric_cols[0] if numeric_cols else df.columns[0])

    # Label-like: non-numeric or string column different from time
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != time_col]
    label_col = label_col or (non_numeric[0] if non_numeric else (df.columns[1] if len(df.columns) > 1 else df.columns[0]))
    return time_col, label_col


def import_data(file_path):
    """Imports the main plot data and normalizes columns to 'Time' and 'Signal'."""
    df = pd.read_csv(file_path)
    time_col, signal_col = detect_time_signal_columns(df)
    df = df[[time_col, signal_col]].copy()
    df.columns = ['Time', 'Signal']
    return df

def import_labelled_peaks(file_path):
    """Imports labelled peaks from a CSV or Excel file and normalizes to columns 'Time' and 'Label'."""
    try:
        if file_path.endswith('.csv'):
            labelled_data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            labelled_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")
    except UnicodeDecodeError:
        labelled_data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Normalize column names: handle lowercase exports (e.g., 'label','time','signal')
    lower_cols = {c.lower(): c for c in labelled_data.columns}
    if 'time' in lower_cols:
        labelled_data.rename(columns={lower_cols['time']: 'Time'}, inplace=True)
    if 'label' in lower_cols:
        labelled_data.rename(columns={lower_cols['label']: 'Label'}, inplace=True)

    # If still missing, try to detect
    if 'Time' not in labelled_data.columns or 'Label' not in labelled_data.columns:
        time_col, label_col = detect_time_label_columns(labelled_data)
        labelled_data.rename(columns={time_col: 'Time', label_col: 'Label'}, inplace=True)

    # Keep only relevant columns if present
    keep_cols = [c for c in ['Label', 'Time', 'Signal'] if c in labelled_data.columns]
    return labelled_data[keep_cols].copy()

def establish_slanted_baseline(peak_data, peak_time):
    """
    Establishes a slanted baseline connecting the left and right troughs
    of the curve around the peak.
    """
    # Find the minimum (trough) on the left and right of the peak
    left_trough_data = peak_data[peak_data['Time'] < peak_time]
    right_trough_data = peak_data[peak_data['Time'] > peak_time]

    if not left_trough_data.empty and not right_trough_data.empty:
        left_trough_time = left_trough_data['Time'][left_trough_data['Signal'] == left_trough_data['Signal'].min()].values[0]
        left_trough_signal = left_trough_data['Signal'].min()

        right_trough_time = right_trough_data['Time'][right_trough_data['Signal'] == right_trough_data['Signal'].min()].values[0]
        right_trough_signal = right_trough_data['Signal'].min()

        return (left_trough_time, left_trough_signal), (right_trough_time, right_trough_signal)
    
    # If no troughs found, return None
    return None


def plot_peak(data, label_row, x_range=30):
    """
    Plots the peak with the label on the curve within a specified x-range around the peak.
    Also attempts to establish and show the slanted baseline.
    """
    peak_time = label_row['Time']
    peak_label = label_row['Label']

    # Define the window around the peak (x_range is the total width of the window)
    start_time = peak_time - x_range / 2
    end_time = peak_time + x_range / 2

    # Filter data within the x-range
    peak_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()

    # Get the y-value (signal) of the peak point
    peak_signal = peak_data.loc[peak_data['Time'].sub(peak_time).abs().idxmin()]['Signal']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(peak_data['Time'], peak_data['Signal'], label='Signal')

    # Annotate the peak
    ax.scatter(peak_time, peak_signal, color='red', label=f'Peak {peak_label}')
    ax.annotate(
        peak_label,
        xy=(peak_time, peak_signal),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        color='red'
    )

    # Customize the plot
    ax.set_xlim([start_time, end_time])
    ax.set_ylim([peak_data['Signal'].min() * 0.9, peak_data['Signal'].max() * 1.1])
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title(f"Peak '{peak_label}' with x-range ~{x_range} units")
    ax.legend()
    ax.grid(True)

    # Step 4: Establish and plot the slanted baseline
    baseline_points = establish_slanted_baseline(peak_data, peak_time)
    auto_baseline = None
    if baseline_points:
        left_trough, right_trough = baseline_points
        auto_baseline = ax.plot([left_trough[0], right_trough[0]], [left_trough[1], right_trough[1]],
                color='green', linestyle='--', label='Slanted Baseline')
        draggable_line = None  # No draggable line needed for slanted baseline
    else:
        # Fallback to draggable baseline at peak signal
        draggable_line = DraggableHorizontalLine(ax, y=peak_signal, color='green', linestyle='--')

    return fig, ax, baseline_points, draggable_line, auto_baseline

def calculate_left_shoulder_area(data, left_trough, peak_time):
    """
    Calculate the area for a left shoulder peak.
    The area is between the baseline at the left trough and the curve, 
    from the left trough to the peak, then doubled.
    """
    baseline_y = left_trough[1]

    # Calculate the area between the baseline and the curve from the left trough to the peak
    area = calculate_area(data, lambda t: baseline_y, left_trough[0], peak_time)
    
    # Double the area for the left shoulder calculation
    return area * 2


def calculate_right_shoulder_area(data, right_trough, peak_time):
    """
    Calculate the area for a right shoulder peak.
    The area is between the baseline at the right trough and the curve, 
    from the peak to the right trough, then doubled.
    """
    baseline_y = right_trough[1]

    # Calculate the area between the baseline and the curve from the peak to the right trough
    area = calculate_area(data, lambda t: baseline_y, peak_time, right_trough[0])
    
    # Double the area for the right shoulder calculation
    return area * 2

def draggable_baseline(ax, initial_baseline):
    """Adds a draggable horizontal baseline."""
    draggable_line = DraggableHorizontalLine(ax, y=initial_baseline)
    return draggable_line

def calculate_area(data, baseline_at_time, left_boundary, right_boundary):
    """Calculates the area between the curve and the slanted baseline."""
    # Filter the data within the integration range
    integration_data = data[(data['Time'] >= left_boundary) & (data['Time'] <= right_boundary)]

    # Calculate the signal above the time-dependent baseline
    times = integration_data['Time'].values
    signals = integration_data['Signal'].values

    # Compute the baseline at each time point
    baseline_values = np.array([baseline_at_time(t) for t in times])

    # Calculate the signal above the baseline
    signal_above_baseline = signals - baseline_values

    # Ignore negative values (below the baseline)
    signal_above_baseline[signal_above_baseline < 0] = 0

    # Calculate the area under the curve using the trapezoidal rule
    area = np.trapz(signal_above_baseline, times)
    
    return area

def export_to_excel(areas, output_file):
    """Exports the calculated areas along with peak labels and coordinates to an Excel file."""
    df = pd.DataFrame(areas)
    df.to_excel(output_file, index=False)

def remove_fills(ax):
    # Remove all PolyCollections (fills) from the Axes
    for artist in ax.get_children():
        if isinstance(artist, PolyCollection):
            artist.remove()

def compute_gmm_peak_area(peak_data: pd.DataFrame, peak_time: float, baseline_function=None, max_components: int = 3):
    """
    Compute peak area using Gaussian Mixture responsibilities to apportion
    baseline-corrected signal among components. Returns area for the
    component whose mean is closest to peak_time. Falls back to None if
    scikit-learn is unavailable or data is insufficient.
    """
    if GaussianMixture is None:
        return None
    if peak_data is None or peak_data.empty:
        return None

    times = peak_data['Time'].values
    signals = peak_data['Signal'].values

    # Baseline correction
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0

    # If almost no signal, skip
    if np.nanmax(corr) <= 0 or np.allclose(corr, 0):
        return 0.0

    X = times.reshape(-1, 1)

    # Choose number of components by BIC (1..max_components)
    best_gmm = None
    best_bic = np.inf
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
            gmm.fit(X, sample_weight=corr)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue

    if best_gmm is None:
        return None

    # Responsibilities
    try:
        resp = best_gmm.predict_proba(X)
    except Exception:
        return None

    # Find component closest to peak_time
    means = best_gmm.means_.flatten()
    target_idx = int(np.argmin(np.abs(means - peak_time)))

    # Area apportionment using responsibilities as weights
    y_weighted = corr * resp[:, target_idx]
    try:
        area = np.trapz(y_weighted, times)
    except Exception:
        # Fallback simple sum approximation
        area = float(np.sum(y_weighted))
    return float(area)

def compute_gmm_component_curve(peak_data: pd.DataFrame, peak_time: float, baseline_function=None, max_components: int = 3):
    """
    Fit a Gaussian Mixture to the baseline-corrected signal (corr) and
    return a smooth Gaussian component curve for the component whose mean
    is nearest to peak_time. The curve is scaled so that its integral equals
    the component's share of the total area under corr.

    Returns (times, component_curve, area). If GMM is unavailable, returns
    (None, None, None).
    """
    if GaussianMixture is None or peak_data is None or peak_data.empty:
        return None, None, None

    times = peak_data['Time'].values
    signals = peak_data['Signal'].values
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0
    total_area = float(np.trapz(corr, times))
    if total_area <= 0 or np.allclose(corr, 0):
        return times, np.zeros_like(corr), 0.0

    X = times.reshape(-1, 1)
    best_gmm = None
    best_bic = np.inf
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
            gmm.fit(X, sample_weight=corr)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue
    if best_gmm is None:
        return None, None, None

    weights = best_gmm.weights_.flatten()
    means = best_gmm.means_.flatten()
    covs = best_gmm.covariances_
    # Extract sigmas for 1D
    sigmas = []
    for c in covs:
        if np.ndim(c) == 0:
            var = float(c)
        else:
            var = float(np.squeeze(c))
        sigmas.append(np.sqrt(max(var, 1e-12)))
    sigmas = np.array(sigmas)

    target_idx = int(np.argmin(np.abs(means - peak_time)))
    # Component area is share of total area
    comp_area = total_area * float(weights[target_idx])
    mu = float(means[target_idx])
    sigma = float(sigmas[target_idx])
    # Gaussian PDF
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    pdf = norm * np.exp(-0.5 * ((times - mu) / sigma) ** 2)
    component_curve = comp_area * pdf  # integrates to comp_area
    return times, component_curve, float(comp_area)

def compute_moment_gaussian_curve(peak_data: pd.DataFrame, baseline_function=None):
    """
    Fallback: derive a single Gaussian from the first two moments of the
    baseline-corrected signal. Ensures the Gaussian area equals the total
    area of the corrected signal within the window.

    Returns (times, gaussian_curve, area, mu, sigma)
    """
    times = peak_data['Time'].values
    signals = peak_data['Signal'].values
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0

    area = float(np.trapz(corr, times))
    if area <= 0 or np.allclose(corr, 0):
        return times, np.zeros_like(corr), 0.0, float(times.mean()), 1.0

    # Weighted mean and variance using corr as weights
    w = corr
    mu = float(np.sum(w * times) / (np.sum(w) + 1e-12))
    var = float(np.sum(w * (times - mu) ** 2) / (np.sum(w) + 1e-12))
    sigma = float(np.sqrt(max(var, 1e-12)))

    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    pdf = norm * np.exp(-0.5 * ((times - mu) / sigma) ** 2)
    gaussian_curve = area * pdf
    return times, gaussian_curve, area, mu, sigma

def compute_gmm_least_squares_component_curve(peak_data: pd.DataFrame, peak_time: float, baseline_function=None, max_components: int = 3):
    """
    Fit a small Gaussian mixture to the baseline-corrected signal using
    GMM to seed means/variances, then solve for non-negative amplitudes via
    linear least squares so the mixture closely matches the curve.

    Returns (times, comp_curve, comp_area) for the component whose mean is
    closest to peak_time. Falls back to None if fitting fails.
    """
    times = peak_data['Time'].values
    signals = peak_data['Signal'].values
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0
    if np.nanmax(corr) <= 0 or np.allclose(corr, 0):
        return None, None, None

    if GaussianMixture is None:
        return None, None, None

    X = times.reshape(-1, 1)
    best_gmm = None
    best_bic = np.inf
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
            gmm.fit(X, sample_weight=corr)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue
    if best_gmm is None:
        return None, None, None

    means = best_gmm.means_.flatten()
    covs = best_gmm.covariances_
    sigmas = []
    for c in covs:
        if np.ndim(c) == 0:
            var = float(c)
        else:
            var = float(np.squeeze(c))
        sigmas.append(np.sqrt(max(var, 1e-12)))
    sigmas = np.array(sigmas)

    # Build design matrix of normalized Gaussians (area 1)
    def gaussian_column(mu, sigma):
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        return norm * np.exp(-0.5 * ((times - mu) / sigma) ** 2)

    Phi = np.column_stack([gaussian_column(mu, sig) for mu, sig in zip(means, sigmas)])

    # Solve for amplitudes (areas) using least squares; enforce non-negativity crudely
    try:
        amps, *_ = np.linalg.lstsq(Phi, corr, rcond=None)
        amps = np.where(amps < 0, 0, amps)
        # Optional refinement: refit with only positive columns
        pos_idx = np.where(amps > 0)[0]
        if len(pos_idx) > 0 and len(pos_idx) < len(amps):
            Phi_pos = Phi[:, pos_idx]
            amps_pos, *_ = np.linalg.lstsq(Phi_pos, corr, rcond=None)
            amps_pos = np.where(amps_pos < 0, 0, amps_pos)
            amps[:] = 0
            amps[pos_idx] = amps_pos
    except Exception:
        return None, None, None

    # Select component closest to peak_time
    target_idx = int(np.argmin(np.abs(means - peak_time)))
    comp_curve = Phi[:, target_idx] * float(amps[target_idx])
    comp_area = float(amps[target_idx])  # since column area is 1
    return times, comp_curve, comp_area

def compute_two_component_gaussian_curve(peak_data: pd.DataFrame, peak_time: float, baseline_function=None, max_components: int = 3):
    """
    Build a two-Gaussian approximation around the peak using GMM seeded
    means/sigmas and non-negative least squares to solve amplitudes.

    Returns (times, combined_curve, total_area, (mu1, sigma1, area1), (mu2, sigma2, area2)).
    On failure, returns (None, None, None, None, None).
    """
    times = peak_data['Time'].values
    signals = peak_data['Signal'].values
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0
    if np.nanmax(corr) <= 0 or np.allclose(corr, 0):
        return None, None, None, None, None

    if GaussianMixture is None:
        return None, None, None, None, None

    X = times.reshape(-1, 1)
    best_gmm = None
    best_bic = np.inf
    for n in range(2, max_components + 1):  # need at least 2
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
            gmm.fit(X, sample_weight=corr)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue
    if best_gmm is None or best_gmm.n_components < 2:
        return None, None, None, None, None

    means = best_gmm.means_.flatten()
    covs = best_gmm.covariances_
    sigmas = []
    for c in covs:
        if np.ndim(c) == 0:
            var = float(c)
        else:
            var = float(np.squeeze(c))
        sigmas.append(np.sqrt(max(var, 1e-12)))
    sigmas = np.array(sigmas)

    # Pick two closest means to the peak time
    order = np.argsort(np.abs(means - peak_time))
    idx1, idx2 = order[0], order[1]
    mu1, mu2 = float(means[idx1]), float(means[idx2])
    s1, s2 = float(sigmas[idx1]), float(sigmas[idx2])

    def gaussian_column(mu, sigma):
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        return norm * np.exp(-0.5 * ((times - mu) / sigma) ** 2)

    Phi = np.column_stack([gaussian_column(mu1, s1), gaussian_column(mu2, s2)])
    try:
        amps, *_ = np.linalg.lstsq(Phi, corr, rcond=None)
        amps = np.where(amps < 0, 0, amps)
    except Exception:
        return None, None, None, None, None

    area1 = float(amps[0])
    area2 = float(amps[1])
    total_area = area1 + area2
    combined_curve = Phi[:, 0] * area1 + Phi[:, 1] * area2
    return times, combined_curve, total_area, (mu1, s1, area1), (mu2, s2, area2)

def gaussian_columns(times: np.ndarray, mus: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    cols = []
    for mu, sigma in zip(mus, sigmas):
        sigma = float(max(sigma, 1e-12))
        norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        cols.append(norm * np.exp(-0.5 * ((times - mu) / sigma) ** 2))
    return np.column_stack(cols)

def fit_mixture_least_squares(peak_data: pd.DataFrame, baseline_function=None, max_components: int = 5):
    """
    Auto-fit: try mixtures with 1..max_components Gaussians.
    For each n, seed with GMM (weights, means, sigmas), build Gaussian basis
    columns (area=1), and solve non-negative least squares (via clamped lstsq)
    for amplitudes to match baseline-corrected signal. Select the model with
    the lowest penalized RMSE (RMSE * sqrt(1 + 0.15*n)).

    Returns (times, combined_curve, total_area, mus, sigmas, amps).
    """
    times = peak_data['Time'].values
    signals = peak_data['Signal'].values
    if baseline_function is not None:
        baseline_vals = np.array([baseline_function(t) for t in times])
        corr = signals - baseline_vals
    else:
        corr = signals.copy()
    corr[corr < 0] = 0.0
    if np.nanmax(corr) <= 0 or np.allclose(corr, 0):
        return times, np.zeros_like(corr), 0.0, np.array([]), np.array([]), np.array([])

    if GaussianMixture is None:
        # Fallback to single moment gaussian as a degenerate "fit"
        _, g, area, mu, sigma = compute_moment_gaussian_curve(peak_data, baseline_function)
        return times, g, area, np.array([mu]), np.array([sigma]), np.array([area])

    X = times.reshape(-1, 1)
    best = None
    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
            gmm.fit(X, sample_weight=corr)
            means = gmm.means_.flatten()
            covs = gmm.covariances_
            sigmas = []
            for c in covs:
                var = float(np.squeeze(c)) if np.ndim(c) else float(c)
                sigmas.append(np.sqrt(max(var, 1e-12)))
            sigmas = np.array(sigmas)

            Phi = gaussian_columns(times, means, sigmas)
            amps, *_ = np.linalg.lstsq(Phi, corr, rcond=None)
            amps = np.where(amps < 0, 0, amps)
            fit = Phi @ amps
            rmse = float(np.sqrt(np.mean((corr - fit) ** 2)))
            score = rmse * np.sqrt(1.0 + 0.15 * n)
            total_area = float(np.sum(amps))
            if best is None or score < best['score']:
                best = dict(n=n, means=means, sigmas=sigmas, amps=amps, fit=fit, rmse=rmse, score=score, area=total_area)
        except Exception:
            continue

    if best is None:
        _, g, area, mu, sigma = compute_moment_gaussian_curve(peak_data, baseline_function)
        return times, g, area, np.array([mu]), np.array([sigma]), np.array([area])

    return times, best['fit'], best['area'], best['means'], best['sigmas'], best['amps']

def build_global_baseline(data: pd.DataFrame):
    """
    Interactive baseline definition across the entire dataset.
    User drags horizontal spans over floor regions (with no peaks).
    For each span, the minimum signal point within the span is used as a
    baseline anchor point. A piecewise-linear baseline function is created
    through these anchors. Returns (baseline_fn, anchor_points).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Time'], data['Signal'], color='lightgray', label='Signal')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title('Select baseline regions: drag to add, click "Done" when finished')
    ax.grid(True)

    anchor_points = []  # list of (time, signal)
    fills = []
    done = [False]

    def on_select(xmin, xmax):
        if xmin is None or xmax is None:
            return
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        region = data[(data['Time'] >= xmin) & (data['Time'] <= xmax)]
        if region.empty:
            return
        # Choose the minimum in the region as the anchor (floor)
        idx = region['Signal'].idxmin()
        t = float(data.loc[idx, 'Time'])
        y = float(data.loc[idx, 'Signal'])
        anchor_points.append((t, y))
        # Yellow highlight from 0 to y over [xmin, xmax]
        fill = ax.fill_between(region['Time'], 0, region['Signal']*0 + y, color='yellow', alpha=0.3)
        fills.append(fill)
        ax.scatter([t], [y], color='yellow', s=30, zorder=3)
        fig.canvas.draw_idle()

    # Matplotlib 3.3.x expects 'rectprops' instead of 'props'
    span = SpanSelector(ax, on_select, direction='horizontal', useblit=True,
                        rectprops=dict(alpha=0.2, facecolor='yellow'))

    # Buttons
    ax_btn_done = plt.axes([0.81, 0.01, 0.1, 0.05])
    btn_done = Button(ax_btn_done, 'Done Baseline')

    ax_btn_clear = plt.axes([0.70, 0.01, 0.1, 0.05])
    btn_clear = Button(ax_btn_clear, 'Clear')

    def on_done(event):
        done[0] = True

    def on_clear(event):
        anchor_points.clear()
        while fills:
            f = fills.pop()
            try:
                f.remove()
            except Exception:
                pass
        # Redraw without anchors
        ax.collections.clear()
        fig.canvas.draw_idle()

    btn_done.on_clicked(on_done)
    btn_clear.on_clicked(on_clear)

    plt.show(block=False)
    while not done[0]:
        try:
            plt.pause(0.1)
        except Exception:
            break

    plt.close(fig)

    # Sort anchors and ensure at least two points
    anchor_points = sorted(anchor_points, key=lambda p: p[0])
    if len(anchor_points) < 2:
        # Fallback: flat baseline at global min
        y0 = float(data['Signal'].min())
        def baseline_fn(t):
            return y0
        return baseline_fn, [(float(data['Time'].min()), y0), (float(data['Time'].max()), y0)]

    def baseline_fn(t):
        # Piecewise-linear interpolation across anchors with edge extrapolation
        if hasattr(t, '__iter__'):
            return np.array([baseline_fn(tt) for tt in t])
        t = float(t)
        # Left of first
        if t <= anchor_points[0][0]:
            (x0, y0), (x1, y1) = anchor_points[0], anchor_points[1]
            if x1 == x0:
                return y0
            m = (y1 - y0) / (x1 - x0)
            return y0 + m * (t - x0)
        # Right of last
        if t >= anchor_points[-1][0]:
            (x0, y0), (x1, y1) = anchor_points[-2], anchor_points[-1]
            if x1 == x0:
                return y1
            m = (y1 - y0) / (x1 - x0)
            return y1 + m * (t - x1)
        # Between anchors
        for i in range(1, len(anchor_points)):
            x0, y0 = anchor_points[i-1]
            x1, y1 = anchor_points[i]
            if x0 <= t <= x1:
                if x1 == x0:
                    return y0
                m = (y1 - y0) / (x1 - x0)
                return y0 + m * (t - x0)
        return anchor_points[-1][1]

    return baseline_fn, anchor_points

