import os
import logging
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
    # Logging setup
    log_path = os.path.join(os.getcwd(), 'peak_fit.log')
    logger = logging.getLogger('peak_area_v2')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid duplicate handlers on re-run
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    try:
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(fh)
    except Exception:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(sh)
    print(f"Logging to: {log_path}")

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
        logger.info(f"Loaded plot_file={plot_file}, label_file={label_file}, data_rows={len(data)}, labels_rows={len(labels)}")


        areas = []
        baseline_line = None
        peak_line = None
        fill = []

        # Step 2.5: Ask user to define a global baseline once
        print("Define baseline across the full signal window; press Done when finished.")
        baseline_fn, baseline_anchors = build_global_baseline(data)
        try:
            logger.info(f"Baseline anchors count={len(baseline_anchors)} anchors={baseline_anchors[:5]}...")
        except Exception:
            logger.info("Baseline anchors captured.")


        # Prepare a single persistent figure
        fig, ax = plt.subplots(figsize=(10, 6))
        current_area = {'value': None, 'method': 'Classic'}

        def draw_peak(label_row, x_range=30):
            ax.cla()
            peak_time = label_row['Time']
            peak_label = label_row['Label']
            start_time = peak_time - x_range / 2
            end_time = peak_time + x_range / 2
            peak_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)].copy()
            ax.plot(peak_data['Time'], peak_data['Signal'], label='Signal')
            # Draw baseline within window
            t_win = peak_data['Time'].values
            b_win = np.array([baseline_fn(t) for t in t_win])
            ax.plot(t_win, b_win, color='green', linestyle='--', label='Baseline')
            # Annotate peak
            peak_signal = peak_data.loc[peak_data['Time'].sub(peak_time).abs().idxmin()]['Signal']
            ax.scatter(peak_time, peak_signal, color='red', label=f'Peak {peak_label}')
            # Auto-fit full mixture (1..5 components), then fall back
            classic_area = calculate_area(peak_data, baseline_fn, t_win.min(), t_win.max())
            y_fill = peak_data['Signal'].values
            method = 'Classic'
            used_gaussian = False
            try:
                mix_t, mix_curve, mix_area, mus, sigmas, amps = fit_mixture_least_squares(peak_data, baseline_function=baseline_fn, max_components=5)
                if mix_t is not None and np.any(mix_curve > 0):
                    ax.plot(mix_t, b_win + mix_curve, color='orange', linestyle='--', linewidth=2, label=f'Mixture ({len(mus)})')
                    y_fill = b_win + mix_curve
                    current_area['value'] = mix_area
                    method = 'Mixture'
                    used_gaussian = True
                    try:
                        logger.info(f"Auto mixture fit components={len(mus)} area={mix_area:.3f} mus={mus} sigmas={sigmas} amps={amps}")
                    except Exception:
                        pass
            except Exception:
                pass
            if not used_gaussian:
                # Final fallback to a moment-matched single Gaussian
                mg_t, mg_curve, mg_area, mu, sigma = compute_moment_gaussian_curve(peak_data, baseline_function=baseline_fn)
                if mg_area > 0 and np.any(mg_curve > 0):
                    ax.plot(mg_t, b_win + mg_curve, color='orange', linestyle='--', linewidth=2, label='Gaussian (moment)')
                    y_fill = b_win + mg_curve
                    current_area['value'] = mg_area
                    method = 'Gaussian'
                    used_gaussian = True
            if not used_gaussian:
                current_area['value'] = classic_area
            current_area['method'] = method
            # Highlight area
            ax.fill_between(t_win, b_win, y_fill, where=(y_fill >= b_win), color='yellow', alpha=0.35)
            # Area annotation
            try:
                ax.text(0.02, 0.95, f"Area: {current_area['value']:.2f} ({method})", transform=ax.transAxes,
                        ha='left', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                logger.info(f"Initial area method={method} value={current_area['value']:.3f} peak_label={peak_label}")
            except Exception:
                pass
            ax.set_xlim([start_time, end_time])
            ax.set_ylim([peak_data['Signal'].min() * 0.9, peak_data['Signal'].max() * 1.1])
            ax.set_xlabel('Time')
            ax.set_ylabel('Signal')
            ax.set_title(f"Peak '{peak_label}' with x-range ~{x_range} units")
            ax.legend()
            ax.grid(True)
            fig.canvas.draw_idle()
            return peak_data, peak_time, peak_signal

        labels_iter = list(labels.iterrows())
        current_index = 0

        # UI buttons & sliders: confirm, tweak Gaussians if present
        ax_button_area_confirm = plt.axes([0.81, 0.01, 0.1, 0.05])
        button_area_confirm = Button(ax_button_area_confirm, 'Confirm Area')

        # Adjust-to-peak auto-refit button
        ax_button_adjust = plt.axes([0.68, 0.01, 0.12, 0.05])
        button_adjust = Button(ax_button_adjust, 'Adjust to Peak')

        # Sliders for fine control (appear after first draw if gaussian curve exists)
        slider_axes = {
            'mu1': plt.axes([0.15, 0.02, 0.2, 0.02]),
            'sigma1': plt.axes([0.40, 0.02, 0.2, 0.02]),
            'amp1': plt.axes([0.65, 0.02, 0.1, 0.02])
        }
        sliders = {}

        # Draw first peak
        label_idx, label_row = labels_iter[current_index]
        peak_data, peak_time, peak_signal = draw_peak(label_row)
        def compute_and_store_area(label_row, peak_data, peak_time, peak_signal):
            # Use the precomputed area and method from draw_peak
            area = current_area['value']
            areas.append({
                'Label': label_row['Label'],
                'Area': area,
                'Peak Time (X)': peak_time,
                'Peak Signal (Y)': peak_signal,
                'Shoulder': 'global-baseline'
            })

        def on_area_confirm(event):
            nonlocal current_index, peak_data, peak_time, peak_signal
            compute_and_store_area(label_row, peak_data, peak_time, peak_signal)
            try:
                logger.info(f"Confirmed label={label_row['Label']} area={current_area['value']:.3f} method={current_area['method']}")
            except Exception:
                pass
            current_index += 1
            if current_index >= len(labels_iter):
                plt.close(fig)
                return
            _, next_row = labels_iter[current_index]
            pd_next, pt_next, ps_next = draw_peak(next_row)
            # Update closure vars
            peak_data, peak_time, peak_signal = pd_next, pt_next, ps_next
            fig.canvas.draw_idle()

        # Optional: after first draw, if we drew a Gaussian-moment curve, attach sliders to tweak it
        # We only bind sliders to the last plotted orange dashed curve by recomputing on change
        def install_sliders_if_possible():
            try:
                # reconstruct a moment gaussian as a starting point
                mg_t, mg_curve, mg_area, mu, sigma = compute_moment_gaussian_curve(peak_data, baseline_function=baseline_fn)
                if mg_area <= 0:
                    return
                sliders['mu1'] = Slider(slider_axes['mu1'], 'mu', mg_t.min(), mg_t.max(), valinit=mu, valfmt='%.2f')
                sliders['sigma1'] = Slider(slider_axes['sigma1'], 'sigma', max((mg_t.max()-mg_t.min())/200.0, 1e-3), (mg_t.max()-mg_t.min())/2.0, valinit=sigma, valfmt='%.2f')
                sliders['amp1'] = Slider(slider_axes['amp1'], 'amp', 0.0, mg_area*2.0, valinit=mg_area, valfmt='%.0f')

                orange_line = [l for l in ax.get_lines() if l.get_linestyle() == '--' and l.get_color() == 'orange']

                def on_change(val):
                    mu_v = sliders['mu1'].val
                    sigma_v = sliders['sigma1'].val
                    amp_v = sliders['amp1'].val
                    # rebuild gaussian curve
                    norm = 1.0 / (np.sqrt(2.0 * np.pi) * max(sigma_v, 1e-12))
                    g = amp_v * norm * np.exp(-0.5 * ((mg_t - mu_v) / max(sigma_v, 1e-12)) ** 2)
                    # replace or add orange line
                    if orange_line:
                        orange_line[0].set_data(mg_t, np.array([baseline_fn(t) for t in mg_t]) + g)
                    else:
                        ax.plot(mg_t, np.array([baseline_fn(t) for t in mg_t]) + g, color='orange', linestyle='--', linewidth=2, label='Gaussian (manual)')
                    # redraw fill and area label
                    ax.collections = [c for c in ax.collections if c.get_label() != 'area_fill']
                    ax.fill_between(mg_t, np.array([baseline_fn(t) for t in mg_t]), np.array([baseline_fn(t) for t in mg_t]) + g, where=g>=0, color='yellow', alpha=0.35, label='area_fill')
                    current_area['value'] = float(np.trapz(g, mg_t))
                    current_area['method'] = 'manual'
                    for t in ax.texts:
                        t.remove()
                    ax.text(0.02, 0.95, f"Area: {current_area['value']:.2f} (manual)", transform=ax.transAxes,
                            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                    try:
                        logger.info(f"Manual adjust mu={mu_v:.4f} sigma={sigma_v:.4f} amp={amp_v:.3f} area={current_area['value']:.3f}")
                    except Exception:
                        pass
                    fig.canvas.draw_idle()

                sliders['mu1'].on_changed(on_change)
                sliders['sigma1'].on_changed(on_change)
                sliders['amp1'].on_changed(on_change)
            except Exception:
                pass

        install_sliders_if_possible()

        def on_adjust(event):
            # Recompute auto mixture fit on current window and redraw
            nonlocal peak_data, peak_time
            try:
                mix_t, mix_curve, mix_area, mus, sigmas, amps = fit_mixture_least_squares(peak_data, baseline_function=baseline_fn, max_components=5)
                b_vals = np.array([baseline_fn(t) for t in mix_t])
                # update curve
                # Remove previous orange lines
                for line in list(ax.get_lines()):
                    if line.get_linestyle() == '--' and line.get_color() == 'orange':
                        line.remove()
                ax.plot(mix_t, b_vals + mix_curve, color='orange', linestyle='--', linewidth=2, label=f'Mixture ({len(mus)})')
                # update fill
                ax.collections = [c for c in ax.collections if c.get_label() != 'area_fill']
                ax.fill_between(mix_t, b_vals, b_vals + mix_curve, where=mix_curve>=0, color='yellow', alpha=0.35, label='area_fill')
                current_area['value'] = mix_area
                current_area['method'] = 'Mixture'
                for t in list(ax.texts):
                    t.remove()
                ax.text(0.02, 0.95, f"Area: {mix_area:.2f} (Mixture)", transform=ax.transAxes,
                        ha='left', va='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                try:
                    logger.info(f"Adjust button mixture components={len(mus)} area={mix_area:.3f} mus={mus} sigmas={sigmas} amps={amps}")
                except Exception:
                    pass
                fig.canvas.draw_idle()
            except Exception as e:
                logger.exception(f"Adjust to Peak failed: {e}")
            print("Adjust to Peak clicked - check peak_fit.log for details")

        button_adjust.on_clicked(on_adjust)

        button_area_confirm.on_clicked(on_area_confirm)

        plt.show(block=True)


        output_file_name = f"calculated_peak_areas_{os.path.splitext(os.path.basename(plot_file))[0]}.xlsx"
        export_to_excel(areas, output_file_name)
        print(f"Areas have been exported to '{output_file_name}'.")


        # Step 9: Ask user if they want to process another dataset
        response = input("Do you want to process another dataset? (y/n): ").lower()
        if response != 'y':
            continue_process = False


if __name__ == "__main__":
    main()
