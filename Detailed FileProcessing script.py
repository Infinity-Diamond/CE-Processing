import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

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

def detect_encoding(file_path):
    if chardet:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            return result['encoding'] if result['encoding'] else 'latin-1'
    else:
        return 'latin-1'  # Fallback to a broader encoding

def ask_parameters():
    """
    Opens a custom dialog that prompts for processing parameters. 
    The user can either manually input values or click "Run Defaults" to use:
        Start Line: 13, End Line: 10814, Sampling Rate: 4, Y-Axis Minimum: 2000, Y-Axis Maximum: 100000.
    """
    result = {}
    top = tk.Tk()
    top.title("Enter Processing Parameters")
    
    # Labels and Entry widgets with default values pre-inserted
    tk.Label(top, text="Start Line:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
    start_line_entry = tk.Entry(top)
    start_line_entry.insert(0, "13")
    start_line_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(top, text="End Line:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
    end_line_entry = tk.Entry(top)
    end_line_entry.insert(0, "10814")
    end_line_entry.grid(row=1, column=1, padx=5, pady=5)

    tk.Label(top, text="Sampling Rate:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
    sampling_rate_entry = tk.Entry(top)
    sampling_rate_entry.insert(0, "4")
    sampling_rate_entry.grid(row=2, column=1, padx=5, pady=5)

    tk.Label(top, text="Y-Axis Minimum:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
    y_min_entry = tk.Entry(top)
    y_min_entry.insert(0, "2000")
    y_min_entry.grid(row=3, column=1, padx=5, pady=5)

    tk.Label(top, text="Y-Axis Maximum:").grid(row=4, column=0, padx=5, pady=5, sticky="e")
    y_max_entry = tk.Entry(top)
    y_max_entry.insert(0, "100000")
    y_max_entry.grid(row=4, column=1, padx=5, pady=5)

    def run_defaults():
        result["start_line"] = 13
        result["end_line"] = 10814
        result["sampling_rate"] = 4
        result["y_min"] = 2000
        result["y_max"] = 100000
        top.destroy()

    def submit():
        try:
            result["start_line"] = int(start_line_entry.get())
            result["end_line"] = int(end_line_entry.get())
            result["sampling_rate"] = int(sampling_rate_entry.get())
            result["y_min"] = float(y_min_entry.get())
            result["y_max"] = float(y_max_entry.get())
        except Exception as e:
            messagebox.showerror("Invalid Input", "Please ensure all inputs are valid numbers.")
            return
        top.destroy()

    run_defaults_button = tk.Button(top, text="Run Defaults", command=run_defaults)
    run_defaults_button.grid(row=5, column=0, padx=5, pady=10)

    submit_button = tk.Button(top, text="Submit", command=submit)
    submit_button.grid(row=5, column=1, padx=5, pady=10)

    top.mainloop()
    return result

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

    # Create CSV and PNG directories inside the selected input directory
    output_dir_csv = os.path.join(input_dir, "CSV")
    output_dir_plots = os.path.join(input_dir, "PNG")
    os.makedirs(output_dir_csv, exist_ok=True)
    os.makedirs(output_dir_plots, exist_ok=True)

    # Prompt the user for processing parameters with an option to run defaults.
    params = ask_parameters()
    start_line = params["start_line"]
    end_line = params["end_line"]
    sampling_rate = params["sampling_rate"]
    y_min = params["y_min"]
    y_max = params["y_max"]

    process_files(input_dir, output_dir_csv, output_dir_plots, start_line, end_line, sampling_rate, y_min, y_max)
    messagebox.showinfo("Processing Complete", "All files have been processed successfully.")
