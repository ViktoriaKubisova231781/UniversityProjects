# ─── Imports ──────────────────────────────────────────────────────────

import io
import json
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import tifffile
from PIL import Image
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import seaborn as sns
from datetime import datetime
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ─── Analysis functions ───────────────────────────────────────────────

def measure_primary_root_and_tip(mask: np.ndarray
    ) -> Tuple[ float,
                Optional[Tuple[int,int]],
                Optional[Tuple[int,int]],
                float,
                float,
                float,
                float,
                List[Tuple[int,int]] ]:
    """
    Returns:
        (length_px, bottom_tip_coord, top_tip_coord,
         smoothness, angle_deg, depth, span, pixel_path)
    """
    binary = (mask > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if labels.max() == 0:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    i = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, _ = stats[i]
    comp = (labels == i).astype(np.uint8)
    ske = skeletonize(comp[y:y+h, x:x+w].astype(bool)).astype(np.uint8)
    pts = set(map(tuple, np.argwhere(ske)))

    G = nx.Graph()
    for (ry, rx) in pts:
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nb = (ry + dy, rx + dx)
            if nb in pts:
                G.add_edge((ry, rx), nb, weight=np.hypot(dy, dx))
    if G.number_of_nodes() == 0:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    top = min(G.nodes, key=lambda n: n[0])
    bot = max(G.nodes, key=lambda n: n[0])
    try:
        path = nx.dijkstra_path(G, top, bot, 'weight')
        length = nx.dijkstra_path_length(G, top, bot, 'weight')
    except nx.NetworkXNoPath:
        return 0.0, None, None, 0.0, 0.0, 0.0, 0.0, []

    full_path = [(ry + y, rx + x) for (ry, rx) in path]
    top_full = full_path[0]
    bot_full = full_path[-1]

    direct_dist = np.linalg.norm(np.array(bot_full) - np.array(top_full))
    smoothness = round(direct_dist / length, 3) if length > 0 else 0.0

    dy = bot_full[0] - top_full[0]
    dx = bot_full[1] - top_full[1]
    angle_deg = round(math.degrees(math.atan2(dy, dx)), 2)

    depth = abs(dy)
    span = abs(max(p[1] for p in full_path) - min(p[1] for p in full_path))

    return round(length, 2), bot_full, top_full, smoothness, angle_deg, float(depth), float(span), full_path


def adjust_measurements_to_full(
    root_tips:   Dict[str, Tuple[int,int]],
    root_paths:  Dict[str, List[Tuple[int,int]]],
    crop_params: Dict[str, Any]
) -> Tuple[ Dict[str, Tuple[int,int]], Dict[str, List[Tuple[int,int]]] ]:
    """
    Shift tip coords and skeleton paths from the cropped frame back into
    original full‐image coordinates.
    """
    x0 = crop_params['x_start']
    y0 = crop_params['top_crop']

    tips_full  = {}
    paths_full = {}

    for key, tip in root_tips.items():
        if tip is None:
            tips_full[key]  = None
            paths_full[key] = []
        else:
            r, c = tip
            tips_full[key] = (r + y0, c + x0)

            path = root_paths.get(key, [])
            paths_full[key] = [(r0 + y0, c0 + x0) for r0, c0 in path]

    return tips_full, paths_full


def render_full_mask_with_roots_tiff(
    full_mask:     np.ndarray,
    root_lengths:  Dict[str, float],
    tips_full:     Dict[str, Tuple[int,int]],
    paths_full:    Dict[str, List[Tuple[int,int]]]
) -> Tuple[Dict[str, Dict], bytes]:
    # 1) build measurements JSON
    measurements = {
        key: {
            "length_px": float(root_lengths[key]),
            "tip_coord": None if tips_full[key] is None else (int(tips_full[key][0]), int(tips_full[key][1]))
        }
        for key in root_lengths
    }
    desc = json.dumps(measurements)

    # 2) draw onto a Matplotlib figure at full resolution
    h, w = full_mask.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(full_mask, cmap='gray', vmin=0, vmax=255)
    cmap = plt.get_cmap('tab10')

    for idx, key in enumerate(root_lengths):
        # draw the primary path thicker
        path = paths_full.get(key, [])
        if path:
            y_path, x_path = zip(*path)
            ax.plot(x_path, y_path, color=cmap(idx), linewidth=6)

        # draw a big yellow tip and a bold label
        tip = tips_full.get(key)
        if tip:
            ty, tx = tip
            ax.scatter(
                tx, ty,
                s=150,             # bigger marker
                c='yellow',
                edgecolors='black',
                linewidths=2,
                zorder=3
            )
            ax.text(
                tx + 15, ty + 15,
                f"{root_lengths[key]:.1f}px",
                color=cmap(idx),
                fontsize=20,       # large font
                fontweight='bold',
                zorder=4
            )

    ax.axis('off')
    plt.tight_layout(pad=0)

    # 3) grab RGBA buffer
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape((H, W, 4))
    img = buf[:, :, [1,2,3,0]]  # ARGB→RGBA
    plt.close(fig)

    # 4) write TIFF with ImageDescription
    tiff_buf = io.BytesIO()
    tifffile.imwrite(
        tiff_buf,
        img,
        photometric='rgb',
        description=desc
    )
    return measurements, tiff_buf.getvalue()


def overlay_roots_on_image(
    image_bytes: bytes,
    tips_full: Dict[str, Tuple[int,int]],
    paths_full: Dict[str, List[Tuple[int,int]]],
    root_lengths: Dict[str, float]
) -> Tuple[Image.Image, Dict[str,str]]:
    """
    Draws the skeleton paths and tip markers on the original image,
    labels each tip with its length, and also returns a dict of
    measurement strings for API consumers.

    Returns:
      - PIL.Image with overlay
      - Dict mapping plant name -> "<length>px at (row, col)"
    """
    # 1) Load image
    img_arr = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    h, w, _ = img_arr.shape

    # 2) Create a figure exactly the same size in inches
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(img_arr)
    ax.axis("off")

    # 3) Draw each plant
    measurement_strs: Dict[str,str] = {}
    for idx, plant in enumerate(root_lengths):
        path = paths_full.get(plant, [])
        # plot skeleton path
        if path:
            y_coords, x_coords = zip(*path)
            ax.plot(x_coords, y_coords, linewidth=3, label=plant)

        # plot tip and label
        tip = tips_full.get(plant)
        if tip:
            ty, tx = tip
            # bigger yellow tip marker
            ax.scatter(tx, ty, s=150, c="yellow", edgecolors="black", linewidths=2, zorder=3)
            # larger font
            text = f"{root_lengths[plant]:.1f}px"
            ax.text(
                tx + 15, ty + 15, text,
                fontsize=20, fontweight="bold"
            )

            # build the measurement string
            measurement_strs[plant] = f"{root_lengths[plant]:.1f}px at ({ty}, {tx})"

    plt.tight_layout(pad=0)

    # 4) Extract RGB buffer and close figure
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape((h, w, 3))
    plt.close(fig)

    return Image.fromarray(buf), measurement_strs


def build_root_measurement_csv(
    entries: List[Dict[str, any]]
) -> Tuple[str, bytes]:
    """
    Takes a list of measurement dictionaries for each plant in each image
    and builds a CSV file as bytes for download.

    Each entry in entries should contain:
    - filename
    - plant
    - length
    - bottom_tip
    - top_tip
    - smoothness
    - angle
    - depth
    - span

    Returns:
        Tuple of (filename, file_bytes)
    """
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)

    # Header row
    writer.writerow([
        "filename", "plant", "length_px",
        "bottom_tip_y", "bottom_tip_x",
        "top_tip_y", "top_tip_x",
        "smoothness", "angle_deg", "depth", "span"
    ])

    for row in entries:
        writer.writerow([
            row["filename"],
            row["plant"],
            row["length"],
            row["bottom_tip"][0] if row["bottom_tip"] else None,
            row["bottom_tip"][1] if row["bottom_tip"] else None,
            row["top_tip"][0] if row["top_tip"] else None,
            row["top_tip"][1] if row["top_tip"] else None,
            row["smoothness"],
            row["angle"],
            row["depth"],
            row["span"]
        ])

    return ("root_summary.csv", csv_buf.getvalue().encode("utf-8"))

# ─── Report Generation Function ───────────────────────
# ─── Report Generation Function ───────────────────────
def generate_root_analysis_report(
    csv_path: str,
    output_path: str = "full_analysis_report.pdf",
    pink: str = "#ED1E79",
    title: str = "IRIS Analysis Report",
    icon_path: str = r"C:\Users\vikku\OneDrive\Documenten\Buas\2024-25d-fai2-adsai-ViktoriaKubisova231781\Sprint_3\app_frontend\static\iris_logo.png",
    font_family: str = "DejaVu Sans",
    chunk_size: int = 20
):
    """
    Generates a multi-page PDF report from a root measurement CSV, with redesigned formatting.
    """
    # ─── Local Configuration ───────────────────────────

    PINK = pink
    TITLE = title
    ICON = icon_path
    FONT = font_family
    CHUNK_SIZE = chunk_size

    # ─── Load & Normalize Data ─────────────────────────

    df_all = pd.read_csv(csv_path)
    df_all = df_all.rename(columns={'length_px': 'length', 'angle_deg': 'angle'})
    df = df_all[df_all['length'] > 0].copy()

    # ─── Validate Required Columns ──────────────────────

    required_cols = [
        "filename", "plant", "length", "bottom_tip_y", "bottom_tip_x",
        "top_tip_y", "top_tip_x", "smoothness", "angle", "depth", "span"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # ─── Compute Per-Image Summary (preserve CSV order) ──────────────────────
    # Capture original filename order
    original_order = df['filename'].drop_duplicates().tolist()
    summary = (
        df.groupby('filename', sort=False)
          .agg(
             Plant_Count=('plant', 'count'),
             Avg_Length=('length', 'mean'),
             Avg_Angle=('angle', 'mean'),
             Avg_Depth=('depth', 'mean'),
             Avg_Smoothness=('smoothness', 'mean'),
             Avg_Span=('span', 'mean')
          )
          .round(2)
          .reset_index()
    )
    # Reorder summary to match CSV appearance
    summary = summary.set_index('filename').loc[original_order].reset_index()

    # ─── Generate PDF Report ───────────────────────────
    with PdfPages(output_path) as pdf:

        # ─── Page 1: Title & Metadata ──────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor('white')

        # Header ax: logo + title on same line
        axh = fig.add_axes([0, 0.88, 1, 0.12])
        axh.axis('off')

        # Embed logo via AnnotationBbox, increased size
        if os.path.exists(ICON):
            img = mpimg.imread(ICON)
            imagebox = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(imagebox, (0.11, 0.45), frameon=False, xycoords='axes fraction')
            axh.add_artist(ab)

        # Draw title centered on same axis, larger font
        axh.text(0.5, 0.5, TITLE,
                 va='center', ha='center',
                 fontsize=32, fontweight='bold',
                 color=PINK, family=FONT)
        
        # Subheader ax: metadata
        axm = fig.add_axes([0, 0.88, 1, 0.03])
        axm.axis('off')
        meta = f"Date: {datetime.now():%Y-%m-%d}    Images: {summary.shape[0]}"
        axm.text(0.5, 0.5, meta,
                 ha='center', va='center',
                 fontsize=10, family=FONT)

        # Separator before insights
        sep0 = Line2D([0.05, 0.95], [0.88, 0.88],
                      transform=fig.transFigure,
                      color=PINK, linewidth=1)
        fig.add_artist(sep0)

        # ─── Page 1: Key Insights ──────────────────────
        fig.text(0.05, 0.85, 'Key Insights',
                 ha='left', va='center',
                 fontsize=16, fontweight='bold',
                 color=PINK, family=FONT)

        # Bullets block: pink bullets, black text, no inline number coloring
        lmin, lmax = df['length'].min(), df['length'].max()
        alow, ahigh = df['angle'].quantile(0.1), df['angle'].quantile(0.9)
        dmin, dmax = df['depth'].min(), df['depth'].max()
        span_avg = df['span'].mean()
        smo_avg = df['smoothness'].mean()
        sentences = [
            f"Across all samples, root lengths ranged from {lmin:.1f}px to {lmax:.1f}px.",
            f"Most roots exhibited growth angles between {alow:.1f}° and {ahigh:.1f}°.",
            f"Root depths spanned from {dmin:.1f}px to {dmax:.1f}px.",
            f"The average lateral span was {span_avg:.1f}px.",
            f"Mean smoothness was {smo_avg:.2f}, indicating overall straightness."
        ]
        y = 0.815

        for sentence in sentences:
            # draw pink bullet
            fig.text(0.05, y, '•', color=PINK, fontsize=10, family=FONT)
            # draw full sentence in black, expanding as needed
            fig.text(0.07, y, sentence,
                     color='black', fontsize=10, family=FONT)
            y -= 0.025

        # Separator after bullets
        sep1 = Line2D([0.05, 0.95], [y, y],
                      transform=fig.transFigure,
                      color=PINK, linewidth=1)
        fig.add_artist(sep1)

        # ─── Page 1: Mini Metric Distributions (2×3 grid) ────
        gs = fig.add_gridspec(3, 2,
                              left=0.085, right=0.95,
                              bottom=0.05, top=y - 0.03,
                              hspace=0.4, wspace=0.2)
        axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(6)]
        dist_cols = ['Avg_Length', 'Avg_Angle', 'Avg_Depth', 'Avg_Smoothness', 'Avg_Span']
        
        # Five histograms
        for i, metric in enumerate(dist_cols):
            axd = axes[i]
            axd.hist(summary[metric], bins=10, color=PINK,
                     edgecolor='black')
            # Title bold
            axd.set_title(metric.replace('Avg_','Average '), fontsize=10, fontweight='bold')
            # X-axis ticks tilt
            axd.tick_params(axis='x', rotation=35)
            # Y-axis label and integer ticks
            axd.set_ylabel('Number of Images', fontsize=8)
            axd.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Sixth slot: Plant count per image bar chart
        axd = axes[5]
        # Compute integer counts per plant count
        counts = summary['Plant_Count'].value_counts().sort_index()
        # Ensure x-axis uses integer tick labels
        x_vals = counts.index.astype(int)
        axd.bar(x_vals, counts.values, color=PINK, edgecolor='black')
        axd.set_title('Plant Count per Image', fontsize=10, fontweight='bold')
        axd.set_xlabel('Number of Plants', fontsize=8)
        axd.set_ylabel('Number of Images', fontsize=8)
        axd.set_xticks(x_vals)
        axd.tick_params(axis='x', rotation=35)

        # Save Page 1
        pdf.savefig(fig)
        plt.close(fig)

        # ─── Summary Table Pages ────────────────────────────
        for start in range(0, len(summary), CHUNK_SIZE):
            chunk = summary.iloc[start:start+CHUNK_SIZE]
            fig_tab, ax_tab = plt.subplots(figsize=(8.27, 11.69))
            ax_tab.axis('off')

            # Prepare table data
            data = [
                [r['filename'], r['Plant_Count'],
                 f"{r['Avg_Length']:.1f}", f"{r['Avg_Angle']:.1f}",
                 f"{r['Avg_Depth']:.1f}", f"{r['Avg_Smoothness']:.2f}",
                 f"{r['Avg_Span']:.1f}"]
                for _, r in chunk.iterrows()
            ]

            # Column width: filename wider (28%), others share 72%
            colWidths = [0.28] + [0.72 / 6] * 6

            # Dynamically size table height based on rows (including header)
            n_rows = len(data) + 1
            row_height = 0.04
            table_height = min(row_height * n_rows, 0.90)
            table_bottom = 1.0 - table_height

            # Draw table
            table = ax_tab.table(
                cellText=data,
                colLabels=["Filename", "Count", "Length", "Angle", "Depth", "Smoothness", "Span"],
                colWidths=colWidths,
                cellLoc='center', colLoc='center',
                bbox=[0.01, table_bottom, 0.98, table_height]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor(PINK)
                    cell._text.set_color('white')
                else:
                    cell.set_facecolor('white')
                    cell._text.set_color('black')

            # Fixed row heights
            table.scale(1, 1.2)

            # Enable wrapping in filename column
            for (row, col), cell in table.get_celld().items():
                if col == 0:
                    cell._text.set_wrap(True)

            pdf.savefig(fig_tab)
            plt.close(fig_tab)


        # ─── Graph Pages: Visual Summaries ─────────────────
        fig4 = plt.figure(figsize=(8.27, 11.69))
        fig4.patch.set_facecolor('white')

        # Title + separator
        fig4.text(0.5, 0.96, "Visual Summaries",
                  ha='center', va='center',
                  fontsize=15, fontweight='bold',
                  color=PINK, family=FONT)
        
        sep = Line2D([0.05, 0.95], [0.94, 0.94], transform=fig4.transFigure,
                     color=PINK, linewidth=1)
        fig4.add_artist(sep)

        # 3×2 grid under the title
        gs4 = fig4.add_gridspec(3, 2,
                                left=0.1, right=0.95,
                                top=0.9, bottom=0.08,
                                hspace=0.4, wspace=0.3)
        ax = [fig4.add_subplot(gs4[i//2, i%2]) for i in range(6)]

        # 1) Histogram: all root lengths
        ax[0].hist(df['length'], bins=15, color=PINK, edgecolor='black')
        ax[0].set_title("Root Length Distribution", fontsize=10, fontweight='bold')
        ax[0].set_xlabel("Length (px)", fontsize=8); ax[0].set_ylabel("Count", fontsize=8)
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))


        # 2) Histogram: all root angles
        ax[1].hist(df['angle'], bins=15, color=PINK, edgecolor='black')
        ax[1].set_title("Root Angle Distribution", fontsize=10, fontweight='bold')
        ax[1].set_xlabel("Angle (°)", fontsize=8); ax[1].set_ylabel("Count", fontsize=8)
        ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))

        # 3) Scatter: Length vs. Smoothness
        ax[2].scatter(df['length'], df['smoothness'], color=PINK, alpha=0.8)
        ax[2].set_title("Length vs. Smoothness", fontsize=10, fontweight='bold')
        ax[2].set_xlabel("Length (px)", fontsize=8); ax[2].set_ylabel("Smoothness", fontsize=8)

        # 4) Scatter: Span vs. Depth
        ax[3].scatter(df['span'], df['depth'], color=PINK, alpha=0.8)
        ax[3].set_title("Span vs. Depth", fontsize=10, fontweight='bold')
        ax[3].set_xlabel("Span (px)", fontsize=8); ax[3].set_ylabel("Depth (px)", fontsize=8)

        # 5) IQR Outlier Highlight for Angle
        q1, q3 = df['angle'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        ax[4].scatter(df.index, df['angle'], color=PINK, alpha=0.8)
        out = df[(df['angle'] < lower) | (df['angle'] > upper)]
        ax[4].scatter(out.index, out['angle'], color='red')
        ax[4].axhline(lower, linestyle='--', color='purple')
        ax[4].axhline(upper, linestyle='--', color='purple')
        ax[4].set_title("Angle Outliers", fontsize=10, fontweight='bold')
        ax[4].set_xlabel("Sample Index", fontsize=8); ax[4].set_ylabel("Angle (°)", fontsize=8)

        # 6) Correlation heatmap of summary metrics
        # build the corr matrix as before
        corr = summary[['Plant_Count','Avg_Length','Avg_Angle','Avg_Depth','Avg_Smoothness','Avg_Span']].corr()

        # define your “pretty” short names
        short = {
            'Plant_Count':   'Count',
            'Avg_Length':    'Length',
            'Avg_Angle':     'Angle',
            'Avg_Depth':     'Depth',
            'Avg_Smoothness':'Smoothness',
            'Avg_Span':      'Span'
        }

        # rename the index & columns
        corr = corr.rename(index=short, columns=short)

        # pick your pink diverging cmap
        pink_div = sns.diverging_palette(345, 15, s=75, l=50, as_cmap=True)

        # draw it
        sns.heatmap(corr,
                    annot=True, fmt=".2f",
                    cmap=pink_div, cbar=False,
                    ax=ax[5])

        # finally, rotate and pad the tick labels so they never overflow
        ax[5].set_xticklabels(ax[5].get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax[5].set_yticklabels(ax[5].get_yticklabels(), rotation=45, fontsize=8)
        ax[5].set_title("Summary Metrics Correlation", fontsize=10, fontweight='bold')
        
        # Final layout & save
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig4)
        plt.close(fig4)
