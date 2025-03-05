
import os
from helper_functions import *
from DraggableLine import *
from tkinter import Tk, filedialog


def select_csv_file(prompt):
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title=prompt,
        filetypes=[("CSV and Excel files", "*.csv *.xlsx *.xls"), ("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
    )
    root.destroy()
    return file_path

def main():
    continue_process = True
    
    while continue_process:
        # Step 1: Select CSV files for plot and labels
        plot_file = select_csv_file("Select the CSV file for plot data")
        label_file = select_csv_file("Select the CSV file for labels")


        # Check if files are selected
        if not plot_file or not label_file:
            print("Both plot and label files must be selected. Exiting.")
            return


        # Step 2: Import data and labels
        data = import_data(plot_file)
        labels = import_labelled_peaks(label_file)


        areas = []
        baseline_line = None
        peak_line = None
        fill = []


        for _, label_row in labels.iterrows():
            # Step 3: Plot the peak
            fig, ax, baseline_points, draggable_line, auto_baseline = plot_peak(data, label_row, x_range=30)
            peak_time = label_row['Time']


            left_trough, right_trough = baseline_points
            baseline_slope = (right_trough[1] - left_trough[1]) / (right_trough[0] - left_trough[0])


            # Generate baseline values for each time point in the range
            baseline_values = left_trough[1] + baseline_slope * (data['Time'] - left_trough[0])
            # Highlight the area in yellow
            ax.fill_between(data['Time'], data['Signal'], baseline_values,
                            where=(data['Time'] >= left_trough[0]) & (data['Time'] <= right_trough[0]),
                            color='yellow', alpha=0.5)


            peak_signal = data.loc[data['Time'].sub(peak_time).abs().idxmin()]['Signal']


            done_with_peak = False
            baseline_confirmed = [False]
            shoulder_selected = [None]


            while not done_with_peak:
                # Step 5: Set up confirm and shoulder buttons
                ax_button_confirm = plt.axes([0.81, 0.01, 0.1, 0.05])
                button_confirm = Button(ax_button_confirm, 'Confirm')


                ax_button_left_shoulder = plt.axes([0.58, 0.01, 0.1, 0.05])
                ax_button_right_shoulder = plt.axes([0.69, 0.01, 0.1, 0.05])
                ax_button_default = plt.axes([0.40, 0.01, 0.1, 0.05])
                button_left_shoulder = Button(ax_button_left_shoulder, 'Left Shoulder')
                button_right_shoulder = Button(ax_button_right_shoulder, 'Right Shoulder')
                button_default = Button(ax_button_default, 'Back to Default')


                def on_default_clicked(event):
                    shoulder_selected[0] = None
                    while len(ax.get_lines()) > 1:
                        ax.get_lines()[-1].remove()
                    remove_fills(ax)
                    left_trough, right_trough = baseline_points
                    ax.plot([left_trough[0], right_trough[0]], [left_trough[1], right_trough[1]],
                            color='green', linestyle='--', label='Slanted Baseline')


                    ax.fill_between(data['Time'], data['Signal'], baseline_values,
                                    where=(data['Time'] >= left_trough[0]) & (data['Time'] <= right_trough[0]),
                                    color='yellow', alpha=0.5)


                def on_confirm_clicked(event):
                    baseline_confirmed[0] = True


                def on_left_shoulder_clicked(event):
                    shoulder_selected[0] = 'left'
                    while len(ax.get_lines()) > 1:
                        ax.get_lines()[-1].remove()
                    remove_fills(ax)
                    baseline_y = baseline_points[0][1]
                    baseline_line = ax.axhline(y=baseline_y, color='green', linestyle='--', label='Left Shoulder Baseline')
                    peak_line = ax.axvline(x=peak_time, color='blue', linestyle='--', label='Left Peak Line')
                    left_trough_time = baseline_points[0][0]
                    ax.fill_between(data['Time'], data['Signal'], baseline_y,
                                    where=(data['Time'] >= left_trough_time) & (data['Time'] <= peak_time),
                                    color='yellow', alpha=0.5)


                def on_right_shoulder_clicked(event):
                    shoulder_selected[0] = 'right'
                    while len(ax.get_lines()) > 1:
                        ax.get_lines()[-1].remove()
                    remove_fills(ax)
                    baseline_y = baseline_points[1][1]
                    baseline_line = ax.axhline(y=baseline_y, color='green', linestyle='--', label='Right Shoulder Baseline')
                    peak_line = ax.axvline(x=peak_time, color='blue', linestyle='--', label='Right Peak Line')
                    right_trough_time = baseline_points[1][0]
                    ax.fill_between(data['Time'], data['Signal'], baseline_y,
                                    where=(data['Time'] >= peak_time) & (data['Time'] <= right_trough_time),
                                    color='yellow', alpha=0.5)


                button_confirm.on_clicked(on_confirm_clicked)
                button_left_shoulder.on_clicked(on_left_shoulder_clicked)
                button_right_shoulder.on_clicked(on_right_shoulder_clicked)
                button_default.on_clicked(on_default_clicked)


                plt.show(block=False)
                while not baseline_confirmed[0]:
                    plt.pause(0.1)


                if shoulder_selected[0] == 'left':
                    area = calculate_left_shoulder_area(data, baseline_points[0], peak_time)
                    print(f"Left Shoulder Area for peak '{label_row['Label']}': {area}")
                elif shoulder_selected[0] == 'right':
                    area = calculate_right_shoulder_area(data, baseline_points[1], peak_time)
                    print(f"Right Shoulder Area for peak '{label_row['Label']}': {area}")
                elif baseline_confirmed[0]:
                    left_trough, right_trough = baseline_points
                    baseline_slope = (right_trough[1] - left_trough[1]) / (right_trough[0] - left_trough[0])
                    baseline_values = left_trough[1] + baseline_slope * (data['Time'] - left_trough[0])
                    ax.fill_between(data['Time'], data['Signal'], baseline_values,
                                    where=(data['Time'] >= left_trough[0]) & (data['Time'] <= right_trough[0]),
                                    color='yellow', alpha=0.5)
                    area = calculate_area(data, lambda t: left_trough[1] + baseline_slope * (t - left_trough[0]),
                                          left_trough[0], right_trough[0])
                    print(f"Area under peak '{label_row['Label']}': {area}")


                ax_button_area_confirm = plt.axes([0.81, 0.01, 0.1, 0.05])
                button_area_confirm = Button(ax_button_area_confirm, 'Confirm Area')


                area_confirmed = [False]


                def on_area_confirm_clicked(event):
                    area_confirmed[0] = True


                button_area_confirm.on_clicked(on_area_confirm_clicked)
                plt.show(block=False)
                while not area_confirmed[0]:
                    plt.pause(0.1)


                areas.append({
                    'Label': label_row['Label'],
                    'Area': area,
                    'Peak Time (X)': peak_time,
                    'Peak Signal (Y)': peak_signal,
                    'Shoulder': shoulder_selected[0] if shoulder_selected[0] else 'standard'
                })
                done_with_peak = True
                plt.close()


        output_file_name = f"calculated_peak_areas_{os.path.splitext(os.path.basename(plot_file))[0]}.xlsx"
        export_to_excel(areas, output_file_name)
        print(f"Areas have been exported to '{output_file_name}'.")


        # Step 9: Ask user if they want to process another dataset
        response = input("Do you want to process another dataset? (y/n): ").lower()
        if response != 'y':
            continue_process = False


if __name__ == "__main__":
    main()
