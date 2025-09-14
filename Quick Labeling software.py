import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector, RectangleSelector
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

class DataLabelingApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Labeling Tool")

        # Dictionary to hold labels for each file.
        # Keys: absolute file paths; Values: list of dict entries: {"label":, "time":, "signal":}
        self.file_labels = {}

        # ------------------------
        # Overall Layout with PanedWindow (allows adjusting table vs. graph size)
        # ------------------------
        self.paned = tk.PanedWindow(master, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.paned)
        self.right_frame = tk.Frame(self.paned)
        self.paned.add(self.left_frame, minsize=250)
        self.paned.add(self.right_frame)

        # ------------------------
        # Left Frame: File Carousel, Dropdown & Controls
        # ------------------------
        self.carousel_frame = tk.Frame(self.left_frame)
        self.carousel_frame.pack(fill=tk.X, padx=5, pady=5)

        self.prev_button = tk.Button(self.carousel_frame, text="<< Prev", command=self.prev_file)
        self.prev_button.pack(side=tk.LEFT)

        self.current_file_label = tk.Label(self.carousel_frame, text="No file selected", width=30)
        self.current_file_label.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.carousel_frame, text="Next >>", command=self.next_file)
        self.next_button.pack(side=tk.LEFT)

        # Dropdown for file selection
        self.file_dropdown = ttk.Combobox(self.left_frame, state="readonly")
        self.file_dropdown.pack(fill=tk.X, padx=5, pady=5)
        self.file_dropdown.bind("<<ComboboxSelected>>", self.on_file_dropdown)

        # Import button (multiple file selection allowed)
        self.import_button = tk.Button(self.left_frame, text="Import Files", command=self.import_files)
        self.import_button.pack(pady=5)

        # Button to optionally select a section (activates zoom/confirm flow)
        self.select_section_button = tk.Button(self.left_frame, text="Select Section", command=self.start_section_selection)
        self.select_section_button.pack(pady=5)

        # Reset zoom to full view
        self.reset_zoom_button = tk.Button(self.left_frame, text="Reset Zoom", command=self.reset_zoom)
        self.reset_zoom_button.pack(pady=5)

        # Toggle Select Mode (rectangle multiple-peaks selection)
        self.toggle_select_button = tk.Button(self.left_frame, text="Toggle Select Mode", command=self.toggle_select_mode)
        self.toggle_select_button.pack(pady=5)

        # Manual Save Button for current datafile (labels & plot)
        self.save_button = tk.Button(self.left_frame, text="Save File", command=self.save_plot_and_data)
        self.save_button.pack(pady=5)

        # ------------------------
        # Left Frame: Label Table
        # ------------------------
        self.tree = ttk.Treeview(self.left_frame, columns=("Label", "Time", "Signal"), show="headings", height=15)
        self.tree.heading("Label", text="Label")
        self.tree.heading("Time", text="Time")
        self.tree.heading("Signal", text="Signal")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        self.delete_button = tk.Button(self.left_frame, text="Delete Selected Label", command=self.delete_selected_label)
        self.delete_button.pack(pady=5)

        # ------------------------
        # Left Frame: Help Button
        # ------------------------
        self.help_button = tk.Button(self.left_frame, text="Help (h)", command=self.show_help)
        self.help_button.pack(pady=5)

        # ------------------------
        # Right Frame: Embedded Matplotlib Figure
        # ------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_facecolor('black')
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Signal")
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        # Status bar at the bottom of the right frame
        self.status_var = tk.StringVar(value="Ready. Import a file to begin.")
        self.status_label = tk.Label(self.right_frame, textvariable=self.status_var, anchor='w')
        self.status_label.pack(fill=tk.X)

        # ------------------------
        # Global and State Variables
        # ------------------------
        self.df = None              # DataFrame for current file
        self.df_filtered = None     # DataFrame after zoom
        self.file_list = []         # List of imported file paths
        self.current_file_index = 0 # Index for carousel
        self.current_file_path = None
        self.initial_xlim = None
        self.initial_ylim = None

        # For current file: list of placed label dictionaries:
        # Each entry: {"label": label, "time": t, "signal": s, "ann": annotation, "dot": dot, "tree_id": tree id}
        self.placed_labels = []
        self.label_counter = 0      # Counter for labels (alphabetical in point mode, numeric in select mode)
        self.custom_label_mode = False
        self.region_size = 1.0      # Default region width

        # Create a region highlight (vertical rectangle spanning full y-range)
        self.region_square = patches.Rectangle((0, 0), self.region_size, 1, edgecolor='yellow',
                                                 facecolor='yellow', alpha=0.2, lw=1)
        self.ax.add_patch(self.region_square)

        # Mode: "point" for single-click labeling (default) or "select" for rectangle selection
        self.mode = "point"

        # ------------------------
        # Setup Selectors and Event Bindings
        # ------------------------
        # SpanSelector for zooming (for analysis)
        self.span_selector = SpanSelector(self.ax, self.on_span_select, "horizontal", useblit=True,
                                          props=dict(alpha=0.5, facecolor='red'))
        self.span_selector.set_active(False)
        # RectangleSelector for multi-select mode (initially inactive)
        self.rectangle_selector = RectangleSelector(self.ax, self.on_rectangle_select, useblit=True,
                                                      button=[1], interactive=True)
        self.rectangle_selector.set_active(False)

        # Annotation to show next label at the cursor (no region info now)
        self.cursor_label_annotation = self.ax.annotate("", xy=(0, 0), xytext=(10, 10),
                                                          textcoords="offset points", ha='center',
                                                          color='yellow', fontsize=12)
        self.ax.add_artist(self.cursor_label_annotation)

        # Connect events on the canvas
        self.canvas.mpl_connect("motion_notify_event", self.update_cursor_label)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Bind keys to the main window (pressing 'b' cancels rectangle selection and hides the box)
        self.master.bind("<Key>", self.on_key_press)

        # Disable controls that require a file until one is loaded
        self.set_controls_enabled(False)

    # ------------------------
    # File Import and Carousel/Dropdown Functions
    # ------------------------
    def import_files(self):
        file_paths = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
        if file_paths:
            self.file_list = list(file_paths)
        else:
            folder = filedialog.askdirectory(title="Select Data Folder")
            if folder:
                self.file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
        if self.file_list:
            # Update dropdown list with base filenames.
            self.file_dropdown['values'] = [os.path.basename(f) for f in self.file_list]
            self.current_file_index = 0
            self.load_file(self.file_list[self.current_file_index])
            self.file_dropdown.current(self.current_file_index)
            self.set_controls_enabled(True)

    def prev_file(self):
        if self.file_list:
            self.save_current_file_labels()
            self.current_file_index = (self.current_file_index - 1) % len(self.file_list)
            self.load_file(self.file_list[self.current_file_index])
            self.file_dropdown.current(self.current_file_index)

    def next_file(self):
        if self.file_list:
            self.save_current_file_labels()
            self.current_file_index = (self.current_file_index + 1) % len(self.file_list)
            self.load_file(self.file_list[self.current_file_index])
            self.file_dropdown.current(self.current_file_index)

    def on_file_dropdown(self, event):
        idx = self.file_dropdown.current()
        if idx >= 0:
            self.save_current_file_labels()
            self.current_file_index = idx
            self.load_file(self.file_list[self.current_file_index])

    def save_current_file_labels(self):
        # Save current file's label data into self.file_labels
        if self.current_file_path is not None:
            entries = []
            for entry in self.placed_labels:
                entries.append({"label": entry["label"], "time": entry["time"], "signal": entry["signal"]})
            self.file_labels[self.current_file_path] = entries

    def load_file(self, file_path):
        self.current_file_path = file_path
        self.current_file_label.config(text=os.path.basename(file_path))
        raw_df = pd.read_csv(file_path)
        # Detect time and signal columns robustly
        time_col, signal_col = self.detect_time_signal_columns(raw_df)
        df = raw_df[[time_col, signal_col]].copy()
        df.columns = ['Time', 'Signal']
        df['Signal'] = df['Signal'] - df['Signal'].min()
        self.df = df

        # Clear and set up axes without disturbing layout
        self.ax.cla()
        self.ax.set_facecolor('black')
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Signal")
        # Plot original data and store as self.data_line
        self.data_line, = self.ax.plot(self.df['Time'], self.df['Signal'], label="Original Data", color='white')
        self.ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        # Capture initial limits
        self.ax.relim()
        self.ax.autoscale_view()
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

        # Re-add region highlight and cursor annotation
        self.ax.add_patch(self.region_square)
        self.ax.add_artist(self.cursor_label_annotation)

        # Reset current file labeling info and table
        self.placed_labels = []
        self.label_counter = 0
        for item in self.tree.get_children():
            self.tree.delete(item)

        # If there are previously saved labels for this file, restore them
        if file_path in self.file_labels:
            for entry in self.file_labels[file_path]:
                label = entry["label"]
                t = entry["time"]
                s = entry["signal"]
                ann = self.ax.annotate(label, (t, s), textcoords="offset points", xytext=(10, 10),
                                       ha='center', color='yellow')
                dot, = self.ax.plot(t, s, 'ro')
                tree_id = self.tree.insert("", "end", values=(label, t, s))
                self.placed_labels.append({"label": label, "time": t, "signal": s,
                                           "ann": ann, "dot": dot, "tree_id": tree_id})
                self.label_counter += 1

        self.fig.canvas.draw_idle()

        # Reset selectors
        # Do not force section selection on load; user can press "Select Section" if needed
        self.span_selector.set_active(False)
        self.rectangle_selector.set_active(False)
        self.mode = "point"
        self.df_filtered = None
        # Enable/disable carousel buttons
        if len(self.file_list) > 1:
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
        else:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        self.update_status_bar()

    # ------------------------
    # Zooming Relative to Selected Region
    # ------------------------
    def on_span_select(self, xmin, xmax):
        if self.df is not None:
            self.df_filtered = self.df[(self.df['Time'] >= xmin) & (self.df['Time'] <= xmax)]
            # Update the data line with the filtered data.
            self.data_line.set_data(self.df_filtered['Time'], self.df_filtered['Signal'])
            self.ax.set_xlim(xmin, xmax)
            # Adjust y-axis: keep lower bound at 0 and add a 10% margin above the max.
            y_data = self.df_filtered['Signal']
            y_max = y_data.max()
            margin = (y_max * 0.1) if y_max > 0 else 0.1
            self.ax.set_ylim(0, y_max + margin)
            self.fig.canvas.draw_idle()
            if messagebox.askyesno("Confirm Section", f"Is this section okay for labeling?\nTime {xmin:.2f} to {xmax:.2f}?"):
                self.span_selector.set_active(False)
            else:
                self.reset_zoom()
                self.span_selector.set_active(True)
            self.update_status_bar()

    def reset_zoom(self):
        if self.df is not None:
            self.data_line.set_data(self.df['Time'], self.df['Signal'])
            if self.initial_xlim is not None and self.initial_ylim is not None:
                self.ax.set_xlim(self.initial_xlim)
                self.ax.set_ylim(self.initial_ylim)
            else:
                self.ax.set_xlim(self.df['Time'].min(), self.df['Time'].max())
                self.ax.relim()
                self.ax.autoscale_view()
            self.fig.canvas.draw_idle()
            # Keep span selector state unchanged here
            self.df_filtered = None
            self.update_status_bar()

    def start_section_selection(self):
        # Enable the span selector to let the user drag-select a section
        if self.df is None:
            messagebox.showinfo("Select Section", "Import a file first, then try again.")
            return
        # Reset any previous filter view to initial range before selecting
        self.data_line.set_data(self.df['Time'], self.df['Signal'])
        if self.initial_xlim is not None and self.initial_ylim is not None:
            self.ax.set_xlim(self.initial_xlim)
            self.ax.set_ylim(self.initial_ylim)
        else:
            self.ax.set_xlim(self.df['Time'].min(), self.df['Time'].max())
            self.ax.relim()
            self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.df_filtered = None
        self.span_selector.set_active(True)
        messagebox.showinfo("Select Section", "Drag horizontally on the plot to choose a section, then confirm.")
        self.update_status_bar()

    def toggle_select_mode(self):
        if self.mode == "point":
            self.mode = "select"
            self.rectangle_selector.set_active(True)
            messagebox.showinfo("Mode Change", "Select mode activated: drag to select multiple peaks.")
        else:
            self.mode = "point"
            self.rectangle_selector.set_active(False)
            messagebox.showinfo("Mode Change", "Point mode activated.")
        self.update_status_bar()

    # ------------------------
    # Redrawing Existing Labels
    # ------------------------
    def redraw_labels(self):
        # Re-create annotations and dots for each label in placed_labels.
        for entry in self.placed_labels:
            label = entry["label"]
            t = entry["time"]
            s = entry["signal"]
            ann = self.ax.annotate(label, (t, s), textcoords="offset points", xytext=(10, 10),
                                   ha='center', color='yellow')
            dot, = self.ax.plot(t, s, 'ro')
            entry["ann"] = ann
            entry["dot"] = dot

    # ------------------------
    # Labeling Functions
    # ------------------------
    def on_click(self, event):
        # In select mode do nothing to prevent unwanted label creation.
        if self.mode != "point":
            return
        if hasattr(event, "inaxes") and event.inaxes == self.ax and event.button == 1:
            x = event.xdata
            # Define region centered at the cursor, width = region_size.
            x_min = x - self.region_size / 2
            x_max = x + self.region_size / 2
            df_to_search = self.df_filtered if self.df_filtered is not None else self.df
            region_df = df_to_search[(df_to_search['Time'] >= x_min) & (df_to_search['Time'] <= x_max)]
            if not region_df.empty:
                max_point = region_df.loc[region_df['Signal'].idxmax()]
                t = max_point['Time']
                s = max_point['Signal']
                if self.custom_label_mode:
                    custom = simpledialog.askstring("Custom Label", "Enter custom label:")
                    label = custom if custom else self.next_label()
                    self.custom_label_mode = False
                else:
                    label = self.next_label()
                ann = self.ax.annotate(label, (t, s), textcoords="offset points", xytext=(10, 10),
                                       ha='center', color='yellow')
                dot, = self.ax.plot(t, s, 'ro')
                tree_id = self.tree.insert("", "end", values=(label, t, s))
                entry = {"label": label, "time": t, "signal": s, "ann": ann, "dot": dot, "tree_id": tree_id}
                self.placed_labels.append(entry)
                self.label_counter += 1
                self.fig.canvas.draw_idle()
                self.update_status_bar()

    def on_rectangle_select(self, eclick, erelease):
        # Only proceed if the selection is large enough.
        if (abs(erelease.xdata - eclick.xdata) < 0.01) or (abs(erelease.ydata - eclick.ydata) < 0.01):
            return
        x_min, x_max = sorted([eclick.xdata, erelease.xdata])
        y_min, y_max = sorted([eclick.ydata, erelease.ydata])
        df_to_search = self.df_filtered if self.df_filtered is not None else self.df
        region_df = df_to_search[(df_to_search['Time'] >= x_min) & (df_to_search['Time'] <= x_max) &
                                  (df_to_search['Signal'] >= y_min) & (df_to_search['Signal'] <= y_max)]
        if not region_df.empty:
            times = region_df['Time'].values
            signals = region_df['Signal'].values
            # Identify local maxima using a simple neighbor comparison.
            peak_indices = []
            for i in range(1, len(signals) - 1):
                if signals[i] > signals[i - 1] and signals[i] > signals[i + 1]:
                    peak_indices.append(i)
            # Sort peaks by time and label them numerically (using current counter).
            for idx, i in enumerate(sorted(peak_indices), start=1):
                t = times[i]
                s = signals[i]
                label = str(self.label_counter + 1)  # numeric label.
                ann = self.ax.annotate(label, (t, s), textcoords="offset points", xytext=(10, 10),
                                       ha='center', color='yellow')
                dot, = self.ax.plot(t, s, 'ro')
                tree_id = self.tree.insert("", "end", values=(label, t, s))
                entry = {"label": label, "time": t, "signal": s, "ann": ann, "dot": dot, "tree_id": tree_id}
                self.placed_labels.append(entry)
                self.label_counter += 1
            self.fig.canvas.draw_idle()
        # After rectangle selection, revert to point mode.
        self.rectangle_selector.set_active(False)
        self.mode = "point"
        self.update_status_bar()

    def next_label(self):
        # In point mode, use alphabetical labels.
        if self.label_counter < 26:
            return chr(65 + self.label_counter)  # 'A' to 'Z'
        else:
            return f"{(self.label_counter // 26) + 1}{chr(65 + (self.label_counter % 26))}"

    def update_cursor_label(self, event):
        if not hasattr(event, "inaxes") or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_center = event.xdata
        next_lab = self.next_label() if not self.custom_label_mode else "Custom"
        self.cursor_label_annotation.xy = (event.xdata, event.ydata)
        self.cursor_label_annotation.set_text(f"Next: {next_lab}")
        self.region_square.set_xy((x_center - self.region_size / 2, self.ax.get_ylim()[0]))
        height = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        self.region_square.set_height(height)
        self.region_square.set_width(self.region_size)
        self.fig.canvas.draw_idle()
        self.update_status_bar()

    def on_scroll(self, event):
        if event.button == 'up':
            self.region_size = min(100, self.region_size * 1.2)
        elif event.button == 'down':
            self.region_size = max(0.1, self.region_size * 0.9)
        self.update_cursor_label(event)

    # ------------------------
    # Key Bindings
    # ------------------------
    def on_key_press(self, event):
        # Pressing 'b' stops displaying the rectangle selection box.
        if event.keysym.lower() == "b":
            if self.mode == "select":
                self.rectangle_selector.set_active(False)
                # Hide the rectangle artist if available.
                if hasattr(self.rectangle_selector, "rect") and self.rectangle_selector.rect is not None:
                    self.rectangle_selector.rect.set_visible(False)
                    self.canvas.draw_idle()
                self.mode = "point"
                return

        key = event.keysym.lower()
        if key == "return":
            self.save_plot_and_data()
        elif key == "c":
            self.label_counter += 1
            self.update_cursor_label(event)
        elif key == "x":
            if self.label_counter > 0:
                self.label_counter -= 1
                self.update_cursor_label(event)
        elif key in ("d", "z"):
            self.undo_last_label()
        elif key == "l":
            self.custom_label_mode = True
            if self.label_counter > 0:
                self.label_counter -= 1
            self.update_cursor_label(event)
        elif key == "m":
            self.toggle_select_mode()
        elif key == "h":
            self.show_help()

    def undo_last_label(self):
        if self.placed_labels:
            entry = self.placed_labels.pop()
            try:
                entry["ann"].remove()
            except Exception:
                entry["ann"].set_visible(False)
            try:
                entry["dot"].remove()
            except Exception:
                pass
            self.tree.delete(entry["tree_id"])
            self.label_counter = max(0, self.label_counter - 1)
            self.fig.canvas.draw_idle()
            self.update_status_bar()

    # ------------------------
    # Table Management and Editing
    # ------------------------
    def delete_selected_label(self):
        selected_items = self.tree.selection()
        if selected_items:
            for item in selected_items:
                for entry in self.placed_labels:
                    if entry["tree_id"] == item:
                        try:
                            entry["ann"].remove()
                        except Exception:
                            entry["ann"].set_visible(False)
                        try:
                            entry["dot"].remove()
                        except Exception:
                            pass
                        self.placed_labels.remove(entry)
                        break
                self.tree.delete(item)
            self.fig.canvas.draw_idle()
            self.update_status_bar()

    def on_tree_double_click(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            current_values = self.tree.item(item, "values")
            new_label = simpledialog.askstring("Edit Label", "Enter new label:", initialvalue=current_values[0])
            if new_label:
                self.tree.item(item, values=(new_label, current_values[1], current_values[2]))
                for entry in self.placed_labels:
                    if entry["tree_id"] == item:
                        entry["label"] = new_label
                        entry["ann"].set_text(new_label)
                        self.fig.canvas.draw_idle()
                        break

    # ------------------------
    # Saving Functionality (using only pandas for Excel)
    # ------------------------
    def save_plot_and_data(self):
        self.save_current_file_labels()
        # Create "LabeledPlots" folder in the current file's directory.
        if self.current_file_path is not None:
            data_folder = os.path.dirname(self.current_file_path)
        else:
            data_folder = os.getcwd()
        save_folder = os.path.join(data_folder, "LabeledPlots")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        # Single prompt for base name; save both PNG and XLSX
        base_filename = filedialog.asksaveasfilename(initialdir=save_folder, defaultextension=".xlsx",
                                                     filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                     title="Save labeled plot and data as...")
        if base_filename:
            try:
                # Ensure we have a base without extension for PNG
                base_root, base_ext = os.path.splitext(base_filename)
                png_path = base_root + ".png"
                xlsx_path = base_root + ".xlsx"

                # Save plot
                self.ax.set_title('')
                leg = self.ax.get_legend()
                if leg is not None:
                    leg.remove()
                self.fig.savefig(png_path, transparent=True)

                # Save data
                if self.current_file_path in self.file_labels:
                    df_labels = pd.DataFrame(self.file_labels[self.current_file_path])
                else:
                    df_labels = pd.DataFrame(columns=["label", "time", "signal"])
                df_labels.to_excel(xlsx_path, index=False)

                messagebox.showinfo("Saved", f"Saved:\n{png_path}\n{xlsx_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save files:\n{e}")

    # ------------------------
    # Help Popup
    # ------------------------
    def show_help(self):
        help_text = (
            "Keyboard Shortcuts:\n"
            "Enter: Save plot and data (creates 'LabeledPlots' folder in data file's directory)\n"
            "c: Skip current letter\n"
            "x: Go back a letter\n"
            "d or z: Undo last label\n"
            "b: Cancel rectangle selection and hide selection box\n"
            "L: Custom label mode\n"
            "m: Toggle between point mode and select mode (for multiple peaks selection)\n"
            "h: Show this help message\n"
            "\n"
            "Other Commands:\n"
            "- Use 'Import Files' to select multiple CSV files (or scan a folder).\n"
            "- Navigate files with the Prev/Next buttons or the dropdown.\n"
            "- In point mode, click on the plot to label the relative maximum within the highlighted region.\n"
            "- Use 'Select Section' to enable horizontal drag selection to zoom into a region for labeling.\n"
            "- Use 'Reset Zoom' to return to the initial full view from import.\n"
            "- Use 'Toggle Select Mode' to switch rectangle selection on/off.\n"
            "- In select mode, drag a rectangle (of sufficient size) to select multiple peaks; they will be labeled numerically (left-to-right).\n"
            "- Press 'b' to cancel rectangle selection and hide the selection box.\n"
            "- Delete a label by selecting it in the table and clicking 'Delete Selected Label'.\n"
            "- Double-click a label in the table to edit it (graph annotation updates accordingly).\n"
        )
        messagebox.showinfo("Help", help_text)

    # ------------------------
    # Utility & Setup
    # ------------------------
    def set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in [self.select_section_button, self.reset_zoom_button, self.toggle_select_button, self.save_button, self.delete_button]:
            btn.config(state=state)
        # Prev/Next managed per file count in load_file

    def update_status_bar(self):
        mode_text = f"Mode: {self.mode}"
        next_lab = self.next_label() if not self.custom_label_mode else "Custom"
        status = f"{mode_text} | Next label: {next_lab} | Region width: {self.region_size:.2f}"
        self.status_var.set(status)

    def detect_time_signal_columns(self, df: pd.DataFrame):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = DataLabelingApp(root)
    root.mainloop()
