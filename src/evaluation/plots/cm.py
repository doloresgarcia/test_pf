import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix
from src.utils.pid_conversion import  pid_conversion_dict, our_to_pandora_mapping, pandora_to_our_mapping
from src.utils.inference.per_particle_metrics import mixed_percentages
import os
from matplotlib.colors import LinearSegmentedColormap
from src.utils.inference.per_particle_metrics import  plot_confusion_matrix, plot_confusion_matrix_pandora
from matplotlib.colors import FuncNorm
from matplotlib.colors import TwoSlopeNorm, PowerNorm







def plot_combined_confusion_matrix(
    sd_hgb1,
    sd_pandora,
    save_dir,
    ax=None,
    suffix="",
    prefix_ml="ML",
    prefix_pandora="Pandora",
    fake_norm="column",
    filename="combined_confusion_matrix_PID.pdf"
):
    """
    Combined confusion matrix:
    - Each cell is split into two vertical halves.
      Left  half: ML 
      Right half: Pandora 
    - Colors encode percentages (using mixed_percentages).
    """

    # ---------------------------
    # 1) Build confusion matrix for ML algorithm 
    # ---------------------------
    class_true_ml = np.array(sd_hgb1["pid_4_class_true"].values, dtype=float)
    class_pred_ml = np.array(sd_hgb1["pred_pid_matched"].values, dtype=float)

    n_classes_true_ml = int(np.nanmax(class_true_ml)) + 1
    n_classes_pred_ml = int(np.nanmax(class_pred_ml)) + 1
    n_classes_ml = max(n_classes_true_ml, n_classes_pred_ml)

    # Index for fake/missed (NaNs in either true or pred)
    class_nan_ml = n_classes_ml

    # Replace NaNs with fake/missed index
    class_true_ml = class_true_ml.copy()
    class_pred_ml = class_pred_ml.copy()
    class_true_ml[np.isnan(class_true_ml)] = class_nan_ml
    class_pred_ml[np.isnan(class_pred_ml)] = class_nan_ml

    labels_ml = list(range(n_classes_ml + 1))
    cm_ml = confusion_matrix(class_true_ml.astype(int),
                             class_pred_ml.astype(int),
                             labels=labels_ml)

    # ---------------------------
    # 2) Build confusion matrix for Pandora (sd_pandora)
    # ---------------------------
    class_true_pan_raw = np.array(sd_pandora.pid.values, dtype=float)
    class_pred_pan_raw = np.array(sd_pandora.pandora_pid.values, dtype=float)

    class_true_no_nan = np.array(
        [pid_conversion_dict[x] for x in class_true_pan_raw[~np.isnan(class_true_pan_raw)]]
    )
    max_class = class_true_no_nan.max()
    
    #max_class = 4
    # map to our class indices, unknown -> max_class+1 ("fake"/"missed")
    class_true_pan = np.array([pid_conversion_dict.get(x, max_class + 1)
                               for x in class_true_pan_raw])
    class_pred_pan = np.array([pandora_to_our_mapping.get(x, max_class + 1)
                               for x in class_pred_pan_raw])

    labels_pan = list(range(int(max(class_true_pan.max(), class_pred_pan.max())) + 1))
    cm_pan = confusion_matrix(class_true_pan.astype(int),
                              class_pred_pan.astype(int),
                              labels=labels_pan)

    # ---------------------------
    # 3) Harmonize number of classes between the two
    # ---------------------------
    n_classes_global = max(cm_ml.shape[0] - 1, cm_pan.shape[0] - 1)  # physical classes
    # we always have +1 row/col for fake/missed at the end
    n_total = n_classes_global + 1

    # Pad ML cm if needed
    if cm_ml.shape[0] < n_total:
        pad_r = n_total - cm_ml.shape[0]
        cm_ml = np.pad(cm_ml, ((0, pad_r), (0, pad_r)))

    # Pad Pandora cm if needed
    if cm_pan.shape[0] < n_total:
        pad_r = n_total - cm_pan.shape[0]
        cm_pan = np.pad(cm_pan, ((0, pad_r), (0, pad_r)))

    # ---------------------------
    # 4) Build class names (same convention as your functions)
    # ---------------------------
    is_muons = (n_classes_global == 5)  # 5 physical: e,CH,NH,gamma,mu

    if is_muons:
        class_names_true = ["e","CH","NH","\u03B3",  "\u03BC", "fake"]
        class_names_pred = ["e","CH","NH","\u03B3", "\u03BC", "missed"]
    else:
        class_names_true = [ "e","CH","NH","\u03B3", "fake"]
        class_names_pred = ["e","CH","NH","\u03B3",   "missed"]

    fake_row = len(class_names_true) - 1  # last row

    # ---------------------------
    # 5) Convert counts to percentages using  mixed_percentages
    # ---------------------------
    cm_ml_percent = mixed_percentages(cm_ml, fake_row, fake_norm=fake_norm)
    cm_pan_percent = mixed_percentages(cm_pan, fake_row, fake_norm=fake_norm)

    # global normalization for color scaling
    vmax = max(np.nanmax(cm_ml_percent), np.nanmax(cm_pan_percent))
    norm = Normalize(vmin=0, vmax=vmax)

    # colormaps:
    cmap_temp = plt.cm.Blues
    cmap = LinearSegmentedColormap.from_list(
    "Blues_truncated",
    cmap_temp(np.linspace(0, 0.7, 256))
)
    
    # ---- plotting part ----
    savefigs = ax is None
    if savefigs:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure


    for i in range(n_total):
        for j in range(n_total):
            p_ml = cm_ml_percent[i, j]
            p_pan = cm_pan_percent[i, j]

            # Lower-left triangle: ML
            if p_ml > 0:
                color_ml = cmap(norm(p_ml))
            else:
                color_ml = (1, 1, 1, 1)

            tri_ml = patches.Polygon(
                [(j,     i + 1),  # bottom-left
                 (j,     i),      # top-left
                 (j + 1, i + 1)], # bottom-right
                closed=True,
                facecolor=color_ml,
                edgecolor="black",
                linewidth=0.3
            )
            ax.add_patch(tri_ml)

            # Upper-right triangle: Pandora
            if p_pan > 0:
                color_pan = cmap(norm(p_pan))
            else:
                color_pan = (1, 1, 1, 1)

            tri_pan = patches.Polygon(
                [(j + 1, i),      # top-right
                 (j,     i),      # top-left
                 (j + 1, i + 1)], # bottom-right
                closed=True,
                facecolor=color_pan,
                edgecolor="black",
                linewidth=0.3
            )
            ax.add_patch(tri_pan)

            # Text annotations: percentages only
        
            ax.text(j + 0.3, i + 0.7, f"{p_ml:.0f}",
                        ha="center", va="center", fontsize=12, color="black")
           
            ax.text(j + 0.7, i + 0.3, f"{p_pan:.0f}",
                        ha="center", va="center", fontsize=12, color="black")

    
    # Bold line separating fake row
    ax.hlines(fake_row, xmin=0, xmax=n_total, linewidth=2, color="black")

    # Ticks & labels
    ax.set_xlim(0, n_total)
    ax.set_ylim(n_total, 0)  # invert y-axis to match usual confusion matrix layout

    ax.set_xticks(np.arange(n_total) + 0.5)
    ax.set_yticks(np.arange(n_total) + 0.5)
    ax.set_xticklabels(class_names_pred)
    ax.set_yticklabels(class_names_true)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("bottom left: HitPF (%), top right: Pandora (%)")

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.tick_params(axis="both", labelsize=14)

    ax.title.set_size(16)



    # --------------------------------------------------------
    # 7)  COLORBAR LEGEND (shared, continuous, percentage)
    # --------------------------------------------------------
    if savefigs:
        cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046, pad=0.04
        )
        cbar.set_label("Percentage (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12) 

        fig.tight_layout()


        out_name = os.path.join(save_dir, filename)
        fig.savefig(out_name, bbox_inches="tight")




#with padding
def plot_combined_confusion_matrix_tb(
    sd_hgb1,
    sd_pandora,
    save_dir,
    ax=None,
    suffix="",
    prefix_ml="ML",
    prefix_pandora="Pandora",
    fake_norm="column",
    filename="combined_confusion_matrix_PID_tb.pdf"
):
    """
    Combined confusion matrix:
    - Each cell is split into two vertical halves.
      Left  half: ML 
      Right half: Pandora 
    - Colors encode percentages (using mixed_percentages).
    """

    # ---------------------------
    # 1) Build confusion matrix for ML algorithm 
    # ---------------------------
    class_true_ml = np.array(sd_hgb1["pid_4_class_true"].values, dtype=float)
    class_pred_ml = np.array(sd_hgb1["pred_pid_matched"].values, dtype=float)

    n_classes_true_ml = int(np.nanmax(class_true_ml)) + 1
    n_classes_pred_ml = int(np.nanmax(class_pred_ml)) + 1
    n_classes_ml = max(n_classes_true_ml, n_classes_pred_ml)

    # Index for fake/missed (NaNs in either true or pred)
    class_nan_ml = n_classes_ml

    # Replace NaNs with fake/missed index
    class_true_ml = class_true_ml.copy()
    class_pred_ml = class_pred_ml.copy()
    class_true_ml[np.isnan(class_true_ml)] = class_nan_ml
    class_pred_ml[np.isnan(class_pred_ml)] = class_nan_ml

    labels_ml = list(range(n_classes_ml + 1))
    cm_ml = confusion_matrix(class_true_ml.astype(int),
                             class_pred_ml.astype(int),
                             labels=labels_ml)

    # ---------------------------
    # 2) Build confusion matrix for Pandora (sd_pandora)
    # ---------------------------
    class_true_pan_raw = np.array(sd_pandora.pid.values, dtype=float)
    class_pred_pan_raw = np.array(sd_pandora.pandora_pid.values, dtype=float)

    class_true_no_nan = np.array(
        [pid_conversion_dict[x] for x in class_true_pan_raw[~np.isnan(class_true_pan_raw)]]
    )
    max_class = class_true_no_nan.max()
    
    #max_class = 4
    # map to our class indices, unknown -> max_class+1 ("fake"/"missed")
    class_true_pan = np.array([pid_conversion_dict.get(x, max_class + 1)
                               for x in class_true_pan_raw])
    class_pred_pan = np.array([pandora_to_our_mapping.get(x, max_class + 1)
                               for x in class_pred_pan_raw])

    labels_pan = list(range(int(max(class_true_pan.max(), class_pred_pan.max())) + 1))
    cm_pan = confusion_matrix(class_true_pan.astype(int),
                              class_pred_pan.astype(int),
                              labels=labels_pan)

    # ---------------------------
    # 3) Harmonize number of classes between the two
    # ---------------------------
    n_classes_global = max(cm_ml.shape[0] - 1, cm_pan.shape[0] - 1)  # physical classes
    # we always have +1 row/col for fake/missed at the end
    n_total = n_classes_global + 1

    # Pad ML cm if needed
    if cm_ml.shape[0] < n_total:
        pad_r = n_total - cm_ml.shape[0]
        cm_ml = np.pad(cm_ml, ((0, pad_r), (0, pad_r)))

    # Pad Pandora cm if needed
    if cm_pan.shape[0] < n_total:
        pad_r = n_total - cm_pan.shape[0]
        cm_pan = np.pad(cm_pan, ((0, pad_r), (0, pad_r)))

    effi_ml = np.trace(cm_ml) / np.sum(cm_ml)
    effi_pan = np.trace(cm_pan) / np.sum(cm_pan)

    print("effi ml")
    print(effi_ml)
    print("pan")
    print(effi_pan)

    # ---------------------------
    # 4) Build class names (same convention as your functions)
    # ---------------------------
    is_muons = (n_classes_global == 5)  # 5 physical: e,CH,NH,gamma,mu

    if is_muons:
        display_order = [1, 3, 2, 0, 4, 5]
        #class_names_true = ["e","CH","NH","\u03B3",  "\u03BC", "fake"]
        #class_names_pred = ["e","CH","NH","\u03B3", "\u03BC", "missed"]
        class_names_true = ["CH","$\gamma$","NH","e", "$\mu$", "fake"]
        class_names_pred = ["CH","$\gamma$","NH","e", "$\mu$", "missed"]
    else:
        display_order = [1, 3, 2, 0, 4]
        class_names_true = [ "CH","$\gamma$","NH","e", "fake"]
        class_names_pred = ["CH","$\gamma$","NH","e", "missed"]

    fake_row = len(class_names_true) - 1  # last row

    cm_ml = cm_ml[np.ix_(display_order, display_order)]
    cm_pan = cm_pan[np.ix_(display_order, display_order)]

    # ---------------------------
    # 5) Convert counts to percentages using  mixed_percentages
    # ---------------------------
    
    cm_ml_percent = mixed_percentages(cm_ml, fake_row, fake_norm=fake_norm)
    cm_pan_percent = mixed_percentages(cm_pan, fake_row, fake_norm=fake_norm)
    
    # global normalization for color scaling
    vmax = max(np.nanmax(cm_ml_percent), np.nanmax(cm_pan_percent))
    norm = Normalize(vmin=0, vmax=vmax)
    cmap_temp = plt.cm.Blues

  
    cmap = LinearSegmentedColormap.from_list(
    "Blues_truncated",
    cmap_temp(np.linspace(0.0, 0.6, 256))
    #cmap_temp(np.linspace(0.00, 0.60, 256))
    #cmap_temp(np.linspace(0, 0.7, 256))
    )

  

    
    # ---- plotting part ----
    savefigs = ax is None
    if savefigs:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    pad = 0.04  # inner padding inside each tile
    extra_gap = 0.2

    # --- vertical gap between mu and fake row ---
    # fake is always last row in your class_names_true
    if "fake" in class_names_true:
        gap_row = class_names_true.index("fake")
    else:
        gap_row = len(class_names_true) - 1  # last row as fallback

    row_gap = 0.3  # size of vertical gap between μ and fake

    def row_offset(i: int) -> float:
        """
        Extra offset for row i. All rows from 'fake' downward
        are shifted by row_gap, creating a larger gap between μ and fake.
        """
        return 0.0 if i < gap_row else row_gap

    for i in range(n_total):
        off = row_offset(i)

        # vertical coordinates (with row offset and inner padding)
        y_top    = i + off + pad
        y_bottom = i + 1 + off - pad
        h = y_bottom - y_top
        half_h = h / 2.0

        for j in range(n_total):
            p_ml  = cm_ml_percent[i, j]   # HitPF (TOP)
            p_pan = cm_pan_percent[i, j]  # Pandora (BOTTOM)
            

            # Colors
            p_ml_plot  = 0.0 if np.isnan(p_ml)  else p_ml
            p_pan_plot = 0.0 if np.isnan(p_pan) else p_pan

    
            color_ml  = cmap(norm(p_ml_plot))
            color_pan = cmap(norm(p_pan_plot))

            # horizontal coordinates (no column gap, just padding)
            x_left  = j + pad
            x_right = j + 1 - pad
            w = x_right - x_left

            # TOP rectangle: HitPF / ML
            rect_ml = patches.Rectangle(
                (x_left, y_top),
                w,
                half_h,
                facecolor=color_ml,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.add_patch(rect_ml)

            # BOTTOM rectangle: Pandora
            rect_pan = patches.Rectangle(
                (x_left, y_top + half_h),
                w,
                half_h,
                facecolor=color_pan,
                edgecolor="black",
                linewidth=0.3,
            )
            ax.add_patch(rect_pan)

            # --- TEXT ---
            x_center = 0.5 * (x_left + x_right)
            y_center_top    = y_top + 0.5 * half_h
            y_center_bottom = y_top + 1.5 * half_h

            # HitPF (TOP)
            ax.text(
                x_center,
                y_center_top,
                f"{p_ml:.0f}",
                ha="center", va="center",
                fontsize=17,
                #color=(color_hi if (not hi_is_pan) else color_lo),
                #fontweight=("bold" if i==j else "normal")
                color="black"
            )

            # Pandora (BOTTOM)
            ax.text(
                x_center,
                y_center_bottom,
                f"{p_pan:.0f}",
                ha="center", va="center",
                fontsize=17, 
                #color=(color_hi if hi_is_pan else color_lo),
                #fontweight=("bold" if i==j else "normal")
                color="black"
            )

       

    # ---------------------------
    #  Ticks & labels (with row gap)
    # ---------------------------

    # X ticks: uniform, center of each column
    xticks = np.arange(n_total) + 0.5

    # Y ticks: center of each (possibly shifted) row
    yticks = []
    for i in range(n_total):
        off = row_offset(i)
        y_top    = i + off + pad
        y_bottom = i + 1 + off - pad
        yticks.append(0.5 * (y_top + y_bottom))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(class_names_pred)
    ax.set_yticklabels(class_names_true)

    # Limits: use row_y mapping for Y, uniform X
    total_height = (n_total - 1) + row_offset(n_total - 1) + 1
    ax.set_xlim(0, n_total)
    ax.set_ylim(total_height, 0)  # inverted, top row at top


    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("top: HitPF (\%), bottom: Pandora (\%)")

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.tick_params(axis="both", labelsize=16)
    ax.title.set_size(16)

    # remove outer rectangle
    for spine in ax.spines.values():
        spine.set_visible(False)



    
    # Bold line separating fake row
    #ax.hlines(fake_row, xmin=0, xmax=n_total, linewidth=2, color="black")



    # --------------------------------------------------------
    # 7)  COLORBAR LEGEND (shared, continuous, percentage)
    # --------------------------------------------------------
    if savefigs:
        cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046, pad=0.04
        )
        cbar.set_label("Percentage (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12) 

        fig.tight_layout()


        out_name = os.path.join(save_dir, filename)
        fig.savefig(out_name, bbox_inches="tight")

   


def plot_combined_confusion_matrix_tb_lines(
    sd_hgb1,
    sd_pandora,
    save_dir,
    ax=None,
    suffix="",
    prefix_ml="ML",
    prefix_pandora="Pandora",
    fake_norm="column",
    filename="combined_confusion_matrix_PID_tblines_a.pdf"
):
    """
    Combined confusion matrix:
    - Each cell is split into two vertical halves.
      Left  half: ML 
      Right half: Pandora 
    - Colors encode percentages (using mixed_percentages).
    """

    # ---------------------------
    # 1) Build confusion matrix for ML algorithm 
    # ---------------------------
    class_true_ml = np.array(sd_hgb1["pid_4_class_true"].values, dtype=float)
    class_pred_ml = np.array(sd_hgb1["pred_pid_matched"].values, dtype=float)

    n_classes_true_ml = int(np.nanmax(class_true_ml)) + 1
    n_classes_pred_ml = int(np.nanmax(class_pred_ml)) + 1
    n_classes_ml = max(n_classes_true_ml, n_classes_pred_ml)

    # Index for fake/missed (NaNs in either true or pred)
    class_nan_ml = n_classes_ml

    # Replace NaNs with fake/missed index
    class_true_ml = class_true_ml.copy()
    class_pred_ml = class_pred_ml.copy()
    class_true_ml[np.isnan(class_true_ml)] = class_nan_ml
    class_pred_ml[np.isnan(class_pred_ml)] = class_nan_ml

    labels_ml = list(range(n_classes_ml + 1))
    cm_ml = confusion_matrix(class_true_ml.astype(int),
                             class_pred_ml.astype(int),
                             labels=labels_ml)

    # ---------------------------
    # 2) Build confusion matrix for Pandora (sd_pandora)
    # ---------------------------
    class_true_pan_raw = np.array(sd_pandora.pid.values, dtype=float)
    class_pred_pan_raw = np.array(sd_pandora.pandora_pid.values, dtype=float)

    unique_pandora_pids = np.unique(sd_pandora.pandora_pid.values)
    print("unique IDs")
    print(unique_pandora_pids)


    class_true_no_nan = np.array(
        [pid_conversion_dict[x] for x in class_true_pan_raw[~np.isnan(class_true_pan_raw)]]
    )
    max_class = class_true_no_nan.max()
    
    #max_class = 4
    # map to our class indices, unknown -> max_class+1 ("fake"/"missed")
    class_true_pan = np.array([pid_conversion_dict.get(x, max_class + 1)
                               for x in class_true_pan_raw])
    class_pred_pan = np.array([pandora_to_our_mapping.get(x, max_class + 1)
                               for x in class_pred_pan_raw])

    labels_pan = list(range(int(max(class_true_pan.max(), class_pred_pan.max())) + 1))
    cm_pan = confusion_matrix(class_true_pan.astype(int),
                              class_pred_pan.astype(int),
                              labels=labels_pan)

    

    # ---------------------------
    # 3) Harmonize number of classes between the two
    # ---------------------------
    n_classes_global = max(cm_ml.shape[0] - 1, cm_pan.shape[0] - 1)  # physical classes
    # we always have +1 row/col for fake/missed at the end
    n_total = n_classes_global + 1

    # Pad ML cm if needed
    if cm_ml.shape[0] < n_total:
        pad_r = n_total - cm_ml.shape[0]
        cm_ml = np.pad(cm_ml, ((0, pad_r), (0, pad_r)))

    # Pad Pandora cm if needed
    if cm_pan.shape[0] < n_total:
        pad_r = n_total - cm_pan.shape[0]
        cm_pan = np.pad(cm_pan, ((0, pad_r), (0, pad_r)))

    # ---------------------------
    # 4) Build class names (same convention as your functions)
    # ---------------------------
    is_muons = (n_classes_global == 5)  # 5 physical: e,CH,NH,gamma,mu

    if is_muons:
        display_order = [1, 3, 2, 0, 4, 5]
        #class_names_true = ["e","CH","NH","\u03B3",  "\u03BC", "fake"]
        #class_names_pred = ["e","CH","NH","\u03B3", "\u03BC", "missed"]
        class_names_true = ["CH","\u03B3","NH","e", "\u03BC", "fake"]
        class_names_pred = ["CH","\u03B3","NH","e", "\u03BC", "missed"]
    else:
        display_order = [1, 3, 2, 0, 4]
        class_names_true = [ "CH","\u03B3","NH","e", "fake"]
        class_names_pred = ["CH","\u03B3","NH","e", "missed"]

    fake_row = len(class_names_true) - 1  # last row

    cm_ml = cm_ml[np.ix_(display_order, display_order)]
    cm_pan = cm_pan[np.ix_(display_order, display_order)]

    print("invest")
    print(labels_pan)
    print(class_names_true)

    # ---------------------------
    # 5) Convert counts to percentages using  mixed_percentages
    # ---------------------------
    cm_ml_percent = mixed_percentages(cm_ml, fake_row, fake_norm=fake_norm)
    cm_pan_percent = mixed_percentages(cm_pan, fake_row, fake_norm=fake_norm)

    # global normalization for color scaling
    vmax = max(np.nanmax(cm_ml_percent), np.nanmax(cm_pan_percent))
    norm = Normalize(vmin=0, vmax=vmax)

    # colormaps:
    cmap_temp = plt.cm.Blues
    cmap = LinearSegmentedColormap.from_list(
    "Blues_truncated",
    cmap_temp(np.linspace(0, 0.7, 256))
)
  
    # ---- plotting part ----
    savefigs = ax is None
    if savefigs:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # ---- draw underlying confusion-matrix grid ----
    for k in range(n_total + 1):
        # horizontal grid lines
        ax.hlines(k, xmin=0, xmax=n_total,
              colors="gray", linewidth=2)

        # vertical grid lines
        ax.vlines(k, ymin=0, ymax=n_total,
              colors="gray", linewidth=2)



    for i in range(n_total):
        for j in range(n_total):
            p_ml = cm_ml_percent[i, j]
            p_pan = cm_pan_percent[i, j]

            # Colors (white if 0)
            color_ml  = cmap(norm(p_ml))  if p_ml  > 0 else (1, 1, 1, 1)
            color_pan = cmap(norm(p_pan)) if p_pan > 0 else (1, 1, 1, 1)

            # --- geometry for this cell ---
            x_left  = j
            x_right = j + 1
            y_top   = i
            y_bot   = i + 1
            h = y_bot - y_top
            half_h = h / 2.0

            # TOP rectangle: ML
            rect_ml = patches.Rectangle(
            (x_left, y_top),     # bottom-left of top half
            x_right - x_left,    # width
            half_h,              # height
            facecolor=color_ml,
            edgecolor="black",
            linewidth=0.3,
            )
            ax.add_patch(rect_ml)

            # BOTTOM rectangle: Pandora
            rect_pan = patches.Rectangle(
            (x_left, y_top + half_h),   # bottom-left of bottom half
            x_right - x_left,
            half_h,
            facecolor=color_pan,
            edgecolor="black",
            linewidth=0.3,
            )
            ax.add_patch(rect_pan)

            # --- TEXT positions ---
            x_center = 0.5 * (x_left + x_right)
            y_center_top    = y_top + 0.25 * h     # middle of top half
            y_center_bottom = y_top + 0.75 * h     # middle of bottom half

            # ML (TOP)
            ax.text(
            x_center,
            y_center_top,
            f"{p_ml:.0f}",
            ha="center", va="center",
            fontsize=12, color="black",
            )

            # Pandora (BOTTOM)
            ax.text(
            x_center,
            y_center_bottom,
            f"{p_pan:.0f}",
            ha="center", va="center",
            fontsize=12, color="black",
            )

    
    # Bold line separating fake row
    ax.hlines(fake_row, xmin=0, xmax=n_total, linewidth=2, color="black")

    # Ticks & labels
    ax.set_xlim(0, n_total)
    ax.set_ylim(n_total, 0)  # invert y-axis to match usual confusion matrix layout

    ax.set_xticks(np.arange(n_total) + 0.5)
    ax.set_yticks(np.arange(n_total) + 0.5)
    ax.set_xticklabels(class_names_pred)
    ax.set_yticklabels(class_names_true)


    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("top: HitPF (%), bottom: Pandora (%)")

    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.tick_params(axis="both", labelsize=14)

    ax.title.set_size(16)

    for spine in ax.spines.values():
        spine.set_visible(False)



    # --------------------------------------------------------
    # 7)  COLORBAR LEGEND (shared, continuous, percentage)
    # --------------------------------------------------------
    if savefigs:
        cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046, pad=0.04
        )
        cbar.set_label("Percentage (%)", fontsize=14)
        cbar.ax.tick_params(labelsize=12) 

        fig.tight_layout()


        out_name = os.path.join(save_dir, filename)
        fig.savefig(out_name, bbox_inches="tight")




def set_latex_serif_style():
    plt.rc("text", usetex=True)
    #plt.rc("text", usetex=False)
    plt.rc("font", family="serif")
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
        }
    )
    #plt.rcParams.update({
    #    "font.size": 12,
    #    "axes.labelsize": 22,
    #    "mathtext.fontset": "dejavuserif",  # serif math without LaTeX
    #})


def plot_combined_confusion_matrix_energy_split(
    sd_hgb,
    sd_pandora,
    save_dir,
    plotting_style,
    ax=None,
    suffix="",
    prefix_ml="ML",
    prefix_pandora="Pandora",
    fake_norm="column"):

    n_plots = 3
    # set_latex_serif_style()
    fig, ax = plt.subplots(1, 3, figsize=(15*2, 5*2))


    #n_plots = 2
    #set_latex_serif_style()
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    #n_plots = 1
    #set_latex_serif_style()
    #fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    

    energies = [0, 1, 10, 100]
    #energies = [1, 10, 100]
    #energies = [1, 10]

   
    for i in range(len(energies) - 1):
        print("energy")
        print(energies[i])
        ax_i = ax[i]
        #ax_i = ax #for single plot.. don't use for multiple
        cond = ((sd_hgb.true_showers_E > energies[i]) & (sd_hgb.true_showers_E < energies[i + 1])) | (np.isnan(sd_hgb.pid) & ((sd_hgb.pred_showers_E > energies[i]) & (sd_hgb.pred_showers_E < energies[i + 1])))
        sd_hgb_i = sd_hgb[cond]
        cond_pandora = ((sd_pandora.true_showers_E > energies[i]) & (sd_pandora.true_showers_E < energies[i + 1])) |  ((np.isnan(sd_pandora.pid)) & ((sd_pandora.pred_showers_E > energies[i]) & (sd_pandora.pred_showers_E < energies[i + 1])))
        sd_pandora_i = sd_pandora[cond_pandora]
        suffix_i = r"${}\,\mathrm{{GeV}} < \mathrm{{E}} < {}\,\mathrm{{GeV}}$".format(energies[i], energies[i+1])
        #suffix_i = "{} GeV \textless E \textless {} GeV".format(energies[i], energies[i + 1])
        if plotting_style=="tb_padding":
            plot_combined_confusion_matrix_tb(sd_hgb1=sd_hgb_i,sd_pandora=sd_pandora_i,save_dir=save_dir, ax=ax_i, filename="cb_comp_padding.pdf")
        elif plotting_style=="tb_lines":
            print("in here")
            plot_combined_confusion_matrix_tb_lines(sd_hgb1=sd_hgb_i,sd_pandora=sd_pandora_i,save_dir=save_dir,ax=ax_i,  filename="cb_comp_lines.pdf")
        elif plotting_style=="diag":
            plot_combined_confusion_matrix(sd_hgb1=sd_hgb_i,sd_pandora=sd_pandora_i,save_dir=save_dir, ax=ax_i, filename="cb_comp_diag.pdf")

        # optionally: tweak per-axis title to include energy bin
        if energies[i] == 0:
            suffix_i = "$\mathrm{{E}}  < 1\,\mathrm{{GeV}}$"
        ax_i.set_title( f"{suffix_i}" )
      
    
    #set_latex_serif_style()
    fig.tight_layout()
    out_name = os.path.join(save_dir, f"cb_comp_{plotting_style}_energy_split_fontsize_17_updated.pdf")
    fig.savefig(out_name, bbox_inches="tight")




 