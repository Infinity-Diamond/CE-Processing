import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from tkinter import Tk, filedialog, simpledialog, messagebox
import matplotlib.patches as patches

# Global variables for labeling
label_counter = 0
current_label = 'A'
df_filtered = None  # To store the filtered dataframe after zoom
ax = None
fig = None
span_selector = None  # SpanSelector object
cursor_label_annotation = None  # Annotation to show the next label at the cursor
placed_labels = []  # List to store references to placed labels
label_data = []  # List to store label, time, and signal for output to Excel
custom_label_mode = False  # Flag to determine if a custom label is being used
region_size = 1.0  # Initial region size for label selection
region_square = None  # Patch object for the square showing the region

# Function to prompt user to select a CSV file
def select_csv_file():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        load_and_plot_csv(file_path)
    else:
        print("No file selected")

# Function to read the CSV and plot the initial graph
# Function to read the CSV and plot the initial graph
def load_and_plot_csv(file_path):
    global df, ax, fig, span_selector
    df = pd.read_csv(file_path)  # Read the CSV file
    df['Signal'] = df['Signal'] - df['Signal'].min()  # Shift signal to start from 0
    
    # Create the initial plot
    fig, ax = plt.subplots(figsize=(12, 8))  # You can still adjust the figure size as needed

    # Customize the appearance of the plot
    ax.set_facecolor('black')
    ax.plot(df['Time'], df['Signal'], label='Original Data', color='white')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Signal', color='white')
    ax.set_title('Original Data Plot', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    # Set up the SpanSelector for zoom
    span_selector = SpanSelector(ax, on_span_select, 'horizontal', useblit=True, props=dict(alpha=0.5, facecolor='red'))

    # Set up the key event listener to handle labeling shortcuts
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    connect_scroll_event()  # Connect the scroll event to adjust the region size

    # Get the figure manager and set the window size and position
    manager = plt.get_current_fig_manager()
    
    # For TkAgg backend (Tkinter), we can use the following to set geometry
    manager.window.wm_geometry(f"2560x800+0+0")  # Full width, top half height, positioned at the top

    # Show the plot and keep interactive
    plt.show()


# Function to zoom in based on SpanSelector
def on_span_select(xmin, xmax):
    global df_filtered, ax, span_selector
    if not df.empty:
        df_filtered = df[(df['Time'] >= xmin) & (df['Time'] <= xmax)]
        
        # Clear the current axis (but not the figure) and plot the filtered data
        ax.clear()
        ax.set_facecolor('black')
        ax.plot(df_filtered['Time'], df_filtered['Signal'], label='Filtered Data', color='white')
        ax.set_xlabel('Time', color='white')  # Ensure x-axis remains visible
        ax.set_ylabel('Signal', color='white')
        ax.set_title(f'Data from Time {xmin:.2f} to {xmax:.2f}', color='white')
        ax.tick_params(axis='x', colors='white')  # Ensure x-ticks remain visible
        ax.tick_params(axis='y', colors='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        plt.draw()
        
        # Ask if the user wants to proceed with this zoomed section
        root = Tk()
        root.withdraw()
        confirm = messagebox.askyesno("Confirm Section", "Is this section okay for labeling?")
        if confirm:
            # Disable SpanSelector after zoom confirmation
            span_selector.set_active(False)
            enable_labeling()
        else:
            reset_zoom()  # Reset the plot to original if the user doesn't confirm

# Function to reset the zoom if the user doesn't confirm
def reset_zoom():
    global ax, df
    ax.clear()
    ax.set_facecolor('black')
    ax.plot(df['Time'], df['Signal'], label='Original Data', color='white')
    ax.set_xlabel('Time', color='white')  # Ensure x-axis remains visible
    ax.set_ylabel('Signal', color='white')
    ax.set_title('Original Data Plot', color='white')
    ax.tick_params(axis='x', colors='white')  # Ensure x-ticks remain visible
    ax.tick_params(axis='y', colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    plt.draw()

# Function to enable labeling after zoom is confirmed
def enable_labeling():
    global ax, cursor_label_annotation, region_square
    # Connect the click event to the on_click_label function
    fig.canvas.mpl_connect('button_press_event', on_click_label)

    # Track mouse motion to display the next label at the cursor
    fig.canvas.mpl_connect('motion_notify_event', update_cursor_label)
    
    # Create an initial cursor annotation
    cursor_label_annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                                          textcoords="offset points", ha='center', color='yellow', fontsize=12)

    # Add the region square
    region_square = patches.Rectangle((0, 0), region_size, 1, edgecolor='yellow', facecolor='none', lw=1)
    ax.add_patch(region_square)

    print("Labeling enabled. Click on the plot to start labeling.")

# Function to generate the next label
def next_label():
    global label_counter, current_label
    if label_counter < 26:  # From A to Z
        current_label = chr(65 + label_counter)  # 65 is ASCII for 'A'
    else:  # After Z, start 2A, 2B, etc.
        current_label = f"{(label_counter // 26) + 1}{chr(65 + (label_counter % 26))}"
    return current_label

# Function to skip the current letter
def skip_letter():
    global label_counter
    label_counter += 1
    update_cursor_label()

# Function to go back a letter without deleting the last label
def go_back_letter():
    global label_counter
    if label_counter > 0:
        label_counter -= 1
    update_cursor_label()

# Function to delete the last placed label and associated point
def delete_last_label():
    global placed_labels, label_data, label_counter
    if placed_labels:
        last_label, last_dot = placed_labels.pop()  # Remove the last label and dot
        last_label.remove()  # Remove the label annotation
        last_dot.remove()  # Remove the red dot
        label_data.pop()  # Remove the last label data entry
        fig.canvas.draw_idle()
        label_counter -= 1

# Function to update cursor label display as the mouse moves, including region size
def update_cursor_label(event=None):
    global cursor_label_annotation, custom_label_mode, region_size, region_square
    next_label_text = next_label() if not custom_label_mode else "Custom"

    if event and event.xdata is not None and event.ydata is not None:
        cursor_label_annotation.xy = (event.xdata, event.ydata)
        cursor_label_annotation.set_text(f"{next_label_text} (Region: {region_size:.2f})")
# Update the region square position and size
        region_square.set_xy((event.xdata - region_size / 2, ax.get_ylim()[0]))
        region_square.set_width(region_size)
    fig.canvas.draw_idle()

# Function to find the closest data point to a given x (time)
def find_closest_data_point(x):
    global df_filtered
    idx = (df_filtered['Time'] - x).abs().argmin()  # Find the index of the closest time value
    return df_filtered.iloc[idx]['Time'], df_filtered.iloc[idx]['Signal']  # Return the time and signal at that index

# Function to label the highest point within the region around x
# Function to label the highest point within the region around x
def on_click_label(event):
    global label_counter, placed_labels, label_data, custom_label_mode
    if event.inaxes is not None and event.button == 1:  # Left click
        x = event.xdata

        # Find the highest y-value within the region size around the clicked x
        x_min = x - region_size / 2
        x_max = x + region_size / 2
        region_df = df_filtered[(df_filtered['Time'] >= x_min) & (df_filtered['Time'] <= x_max)]

        if not region_df.empty:
            max_point = region_df.loc[region_df['Signal'].idxmax()]
            closest_time, closest_signal = max_point['Time'], max_point['Signal']

            # Capture current axis limits
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()

            if custom_label_mode:
                root = Tk()
                root.withdraw()
                custom_label = simpledialog.askstring("Custom Label", "Enter custom label:")
                label = custom_label if custom_label else next_label()
                custom_label_mode = False  # Reset the custom label mode after usage
            else:
                label = next_label()

            # Add label and dot to the plot
            new_label = ax.annotate(label, (closest_time, closest_signal), textcoords="offset points", xytext=(10, 10),
                                    ha='center', color='yellow')
            new_dot, = ax.plot(closest_time, closest_signal, 'ro')

            label_data.append((label, closest_time, closest_signal))
            placed_labels.append((new_label, new_dot))

            # Restore axis limits
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)

            plt.draw()

            label_counter += 1  # Move to the next label after a click
            update_cursor_label()


# Function to handle key press events (for 'Enter', 'x', 'c', 'd', and 'L')
def on_key_press(event):
    global custom_label_mode
    if event.key == 'enter':
        save_plot_and_data()
    elif event.key == 'c':
        skip_letter()
    elif event.key == 'x':
        go_back_letter()
    elif event.key == 'd':
        delete_last_label()
    elif event.key == 'z':
        delete_last_label()
    elif event.key == 'L':
        custom_label_mode = True
        go_back_letter()
    elif event.key == 's':
        skip_letter()
        go_back_letter()



# Function to save the plot and label data
def save_plot_and_data():
    root = Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save plot as..."
    )
    if file_path:
        ax.set_title('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        fig.savefig(file_path, transparent=True)
        print(f"Plot saved as {file_path} with a transparent background.")

    excel_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        title="Save label data as..."
    )
    if excel_path:
        df_labels = pd.DataFrame(label_data, columns=['Label', 'Time', 'Signal'])
        df_labels.to_excel(excel_path, index=False)
        print(f"Label data saved to {excel_path}.")

# Function to detect the scroll wheel event and adjust region size
def on_scroll(event):
    global region_size
    if event.button == 'up':  # Scroll up to increase region size
        region_size = min(100, region_size * 1.2)  # Limit region size to 100
    elif event.button == 'down':  # Scroll down to decrease region size
        region_size = max(0.1, region_size * 0.9)  # Ensure region size doesn't get too small
    update_cursor_label(event)

# Connect the scroll event to adjust region size
def connect_scroll_event():
    fig.canvas.mpl_connect('scroll_event', on_scroll)

# Start the program by selecting a CSV file
select_csv_file()
