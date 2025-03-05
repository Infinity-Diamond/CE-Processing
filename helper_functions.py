import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from DraggableLine import DraggableHorizontalLine, DraggableVerticalLine
from matplotlib.collections import PolyCollection


def import_data(file_path):
    """Imports the main plot data."""
    return pd.read_csv(file_path)

def import_labelled_peaks(file_path):
    """Imports labelled peaks from a CSV or Excel file."""
    try:
        if file_path.endswith('.csv'):
            # Try reading the CSV file with utf-8 encoding
            labelled_data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            # Read Excel files
            labelled_data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")
    except UnicodeDecodeError:
        # Handle encoding issues by falling back to a different encoding
        labelled_data = pd.read_csv(file_path, encoding='ISO-8859-1')
    
    return labelled_data

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

