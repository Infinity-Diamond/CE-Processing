import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import simpledialog

# Optional: To detect file encoding dynamically
try:
    import chardet
except ImportError:
    chardet = None  # Handle cases where chardet is not installed

def ask_directory(title="Select a Directory", initialdir=os.path.expanduser("~")):
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    if not dir_path:
        messagebox.showwarning("No Directory Selected", f"No directory was selected for '{title}'. Exiting script.")
        exit()
    return dir_path

def ask_integer(title="Input Integer", prompt="Enter an integer:"):
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askinteger(title, prompt, parent=root)
    root.destroy()
    if user_input is None:
        messagebox.showwarning("No Input Provided", f"No input was provided for '{prompt}'. Exiting script.")
        exit()
    return user_input

def ask_float(title="Input Float", prompt="Enter a floating-point number:"):
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askfloat(title, prompt, parent=root)
    root.destroy()
    if user_input is None:
        messagebox.showwarning("No Input Provided", f"No input was provided for '{prompt}'. Exiting script.")
        exit()
    return user_input

def detect_encoding(file_path):
    if chardet:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            return result['encoding'] if result['encoding'] else 'latin-1'
    else:
        return 'latin-1'  # Fallback to a broader encoding

def process_files(input_dir, output_dir_csv, output_dir_plots, start_line=13, end_line=10814, sampling_rate=4, y_min=None, y_max=None):
    # Ensure output directories exist
    os.makedirs(output_dir_csv, exist_ok=True)
    os.makedirs(output_dir_plots, exist_ok=True)

    asc_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".asc")]
    if not asc_files:
        messagebox.showwarning("No .asc Files Found", f"No .asc files were found in {input_dir}. Exiting script.")
        exit()

    for filename in asc_files:
        input_file = os.path.join(input_dir, filename)
        output_file_csv = os.path.join(output_dir_csv, os.path.splitext(filename)[0] + ".csv")
        output_file_plot = os.path.join(output_dir_plots, os.path.splitext(filename)[0] + "_plot.png")

        encoding = detect_encoding(input_file)
        
        try:
            with open(input_file, "r", encoding=encoding, errors='replace') as f:
                lines = f.readlines()
            
            if len(lines) < end_line:
                messagebox.showwarning("File Too Short", f"The file {filename} has fewer lines than expected.")
                continue

            cropped_lines = lines[start_line:end_line]
            cropped_lines = [line.strip() for line in cropped_lines if line.strip()]
            signal = np.array([float(line) for line in cropped_lines])
            time = np.arange(len(signal)) / sampling_rate

            df = pd.DataFrame({"Time": time, "Signal": signal})
            df.to_csv(output_file_csv, index=False)

            # Plotting
            plt.figure(figsize=(10, 4))
            plt.plot(time, signal, linewidth=1)
            
            # Set y-axis limits based on user input
            if y_min is not None and y_max is not None:
                plt.ylim([y_min, y_max])
            
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_file_plot, dpi=300, transparent=True)
            plt.close()

            print(f"Processed and saved: {filename}")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred while processing {filename}:\n{e}")
            continue

if __name__ == "__main__":
    print("Select the input directory containing .asc files:")
    input_dir = ask_directory("Select Input Directory")

    print("Select the directory for CSV outputs:")
    output_dir_csv = ask_directory("Select CSV Output Directory")

    print("Select the directory for plot outputs:")
    output_dir_plots = ask_directory("Select Plot Output Directory")

    # Optional: Ask user for start and end lines and sampling rate
    start_line = ask_integer("Start Line", "Enter the starting line number (default is 13):") or 13
    end_line = ask_integer("End Line", "Enter the ending line number (default is 10814):") or 10814
    sampling_rate = ask_integer("Sampling Rate", "Enter the sampling rate (default is 4 samples/sec):") or 4

    # Ask user for y-axis limits
    y_min = ask_float("Y-Axis Minimum", "Enter the minimum value for the Y-axis (leave blank for auto):")
    y_max = ask_float("Y-Axis Maximum", "Enter the maximum value for the Y-axis (leave blank for auto):")

    process_files(input_dir, output_dir_csv, output_dir_plots, start_line, end_line, sampling_rate, y_min, y_max)

    messagebox.showinfo("Processing Complete", "All files have been processed successfully.")
