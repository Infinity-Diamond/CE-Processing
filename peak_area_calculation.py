import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from DraggableLine import *
from helper_functions import *

# Read the CSV files
data = pd.read_csv('092524_T20.dat.csv')
labels = pd.read_csv('T20_Labelled.csv')

# Adjust the 'Signal' values in labels (add 6709 to align with data)
labels['Signal'] += 6709

# Initialize a list to store the areas
areas = []

# Loop over each peak in the labels DataFrame
for index, row in labels.iterrows():
    peak_label = row['Label']
    peak_time = row['Time']
    peak_signal = row['Signal']
    
    print(f"\nProcessing peak '{peak_label}' at time {peak_time}")
    
    # Define x-range around the peak
    x_range = 30  # Total width of x-range
    start_time = peak_time - x_range / 2
    end_time = peak_time + x_range / 2
    
    # Filter data within x-range
    peak_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()
    
    # Plot the data within x-range
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.set_title(f"Peak '{peak_label}' with x-range ~{x_range} units")
    ax.legend()
    ax.grid(True)
    
    # Initialize variables
    done_with_peak = False
    x_walls_set = False  # Flag to check if x-walls are set
    left_wall = None
    right_wall = None
    
    while not done_with_peak:
        # Add a draggable horizontal line for the baseline
        initial_baseline = peak_data['Signal'].min()
        draggable_line = DraggableHorizontalLine(ax, y=initial_baseline)
        
        # Add buttons: Confirm, Set x-walls
        ax_button_confirm = plt.axes([0.81, 0.01, 0.1, 0.05])  # x, y, width, height
        button_confirm = Button(ax_button_confirm, 'Confirm')
        
        ax_button_xwalls = plt.axes([0.69, 0.01, 0.1, 0.05])
        button_xwalls = Button(ax_button_xwalls, 'Set x-walls')
        
        # Variables to store button states
        baseline_confirmed = [False]
        xwalls_pressed = [False]
        
        def on_button_confirm_clicked(event):
            baseline_confirmed[0] = True
        
        def on_button_xwalls_clicked(event):
            xwalls_pressed[0] = True
        
        button_confirm.on_clicked(on_button_confirm_clicked)
        button_xwalls.on_clicked(on_button_xwalls_clicked)
        
        # Show the plot and wait for user action
        plt.show(block=False)
        while not (baseline_confirmed[0] or xwalls_pressed[0]):
            plt.pause(0.1)
        
        if xwalls_pressed[0]:
            # Remove the Set x-walls button
            button_xwalls.ax.remove()
            plt.draw()
            
            # Add two draggable vertical lines for x-walls
            initial_left_wall = peak_time - 5
            initial_right_wall = peak_time + 5
            draggable_left_wall = DraggableVerticalLine(ax, x=initial_left_wall)
            draggable_right_wall = DraggableVerticalLine(ax, x=initial_right_wall)
            
            # Add the Confirm button (if not already present)
            # Wait for user to click Confirm
            baseline_confirmed = [False]
            while not baseline_confirmed[0]:
                plt.pause(0.1)
            
            # Get the positions of the x-walls
            left_wall = draggable_left_wall.line.get_xdata()[0]
            right_wall = draggable_right_wall.line.get_xdata()[0]
            x_walls_set = True
            
            # Disconnect the x-wall draggable lines
            draggable_left_wall.disconnect()
            draggable_right_wall.disconnect()
            plt.draw()
        else:
            # Proceed with area calculation
            baseline = draggable_line.line.get_ydata()[0]
            print(f"Selected baseline y-value: {baseline}")
            
            # Remove the Confirm and Set x-walls buttons
            button_confirm.ax.remove()
            button_xwalls.ax.remove()
            plt.draw()
            
            # Disconnect the draggable line to prevent further movement
            draggable_line.disconnect()
            
            # Proceed with the rest of the code
            # Subtract the baseline from the signal
            peak_data['Signal_above_baseline'] = peak_data['Signal'] - baseline
            
            if x_walls_set:
                # Use x-walls for integration limits
                left_boundary_time = left_wall
                right_boundary_time = right_wall
                # Ensure left is less than right
                if left_boundary_time > right_boundary_time:
                    left_boundary_time, right_boundary_time = right_boundary_time, left_boundary_time
                # Limit the data to between the x-walls
                integration_data = peak_data[
                    (peak_data['Time'] >= left_boundary_time) & (peak_data['Time'] <= right_boundary_time)
                ]
            else:
                # Find indices where the signal crosses the baseline
                signal_sign = np.sign(peak_data['Signal_above_baseline'])
                sign_changes = np.where(np.diff(signal_sign) != 0)[0]
                
                # Extract times of zero crossings (approximate)
                zero_crossings = peak_data['Time'].values[sign_changes]
                
                # Find the zero crossings to the left and right of the peak
                left_crossings = zero_crossings[zero_crossings < peak_time]
                right_crossings = zero_crossings[zero_crossings > peak_time]
                
                if len(left_crossings) == 0 or len(right_crossings) == 0:
                    print("Could not find crossings on both sides of the peak.")
                    # Remove area highlighting and intersection points if they exist
                    plt.draw()
                    # Provide option to set x-walls
                    x_walls_set = False
                    continue  # Go back to allow user to set x-walls
                else:
                    left_boundary_time = left_crossings[-1]
                    right_boundary_time = right_crossings[0]
                    # Limit the data to between the left and right crossings
                    integration_data = peak_data[
                        (peak_data['Time'] >= left_boundary_time) & (peak_data['Time'] <= right_boundary_time)
                    ]
            
            # Extract time and signal values for integration
            integration_time = integration_data['Time'].values
            integration_signal = integration_data['Signal'].values
            
            # Subtract the baseline from the signal
            signal_above_baseline = integration_signal - baseline
            
            # Only consider positive values (above baseline)
            signal_above_baseline[signal_above_baseline < 0] = 0
            
            # Calculate the area using the trapezoidal rule
            area = np.trapz(signal_above_baseline, integration_time)
            print(f"Area under peak '{peak_label}': {area}")
            
            # Plot the area under the curve
            area_fill = ax.fill_between(
                integration_time,
                integration_signal,
                baseline,
                where=(signal_above_baseline > 0),
                color='yellow',
                alpha=0.5,
                label=f'Area under peak {peak_label}'
            )
            # Mark the boundaries
            if x_walls_set:
                # Draw vertical lines at x-walls
                ax.axvline(x=left_boundary_time, color='blue', linestyle='--', label='Left x-wall')
                ax.axvline(x=right_boundary_time, color='blue', linestyle='--', label='Right x-wall')
            else:
                # Mark the left and right crossings
                intersection_points = ax.scatter(
                    [left_boundary_time, right_boundary_time],
                    [baseline, baseline],
                    color='purple',
                    label='Intersection Points'
                )
            # Update the legend
            ax.legend()
            
            # Redraw the plot with the area highlighted
            plt.draw()
            
            # Add "Retry" and "Next peak" buttons
            retry_pressed = [False]
            next_peak_pressed = [False]
            ax_retry = plt.axes([0.58, 0.01, 0.1, 0.05])
            ax_next = plt.axes([0.69, 0.01, 0.1, 0.05])
            button_retry = Button(ax_retry, 'Retry')
            button_next = Button(ax_next, 'Next peak')
            
            def on_retry_clicked(event):
                retry_pressed[0] = True
            
            def on_next_clicked(event):
                next_peak_pressed[0] = True
            
            button_retry.on_clicked(on_retry_clicked)
            button_next.on_clicked(on_next_clicked)
            
            # Wait for user to click "Retry" or "Next peak"
            while not (retry_pressed[0] or next_peak_pressed[0]):
                plt.pause(0.1)
            
            if retry_pressed[0]:
                # Remove area highlighting and any added lines or points
                area_fill.remove()
                if x_walls_set:
                    # Remove x-walls
                    for line in ax.lines[-2:]:  # Assuming the last two lines are x-walls
                        line.remove()
                else:
                    # Remove intersection points
                    intersection_points.remove()
                # Remove the Retry and Next peak buttons
                button_retry.ax.remove()
                button_next.ax.remove()
                plt.draw()
                # Remove the baseline line
                draggable_line.line.remove()
                plt.draw()
                # Reset flags and variables
                x_walls_set = False
                left_wall = None
                right_wall = None
                # Continue the loop to allow user to adjust baseline and x-walls again
                continue  # Goes back to the start of the while loop
            
            elif next_peak_pressed[0]:
                # Remove the Retry and Next peak buttons
                button_retry.ax.remove()
                button_next.ax.remove()
                plt.draw()
                # Store the area along with the peak label
                areas.append({'Label': peak_label, 'Area': area, 'Time': peak_time, 'Y-value': peak_signal})
                # Close the plot and proceed to next peak
                plt.close()
                done_with_peak = True  # Exit the while loop
    
    # Close the plot if it's still open
    plt.close()

# After processing all peaks, print out the areas
print("\nCalculated Areas for all Peaks:")
for area_info in areas:
    print(f"Peak '{area_info['Label']}': Area = {area_info['Area']}")

# Create a DataFrame from the areas list
areas_df = pd.DataFrame(areas)

# Export the DataFrame to an Excel file
areas_df.to_excel('calculated_peak_areas.xlsx', index=False)
print("\nAreas have been exported to 'calculated_peak_areas.xlsx'.")