import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl


def calculate_agreement_rate(file_path):
    """
    Calculates the agreement rate from a prediction file.

    For each claim in the file, if the prediction starts with "Evet" or "Hayır",
    the first token (split by period) is compared with the expected "agreement" field.
    A warning is printed if the prediction does not contain either token.

    Args:
        file_path (str): Path to the prediction JSON file.

    Returns:
        float: The agreement rate (# agreements / total claims).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"Warning: No results found in file {file_path}")
        return 0.0

    total_claims = 0
    agreements = 0

    for item in results:
        total_claims += 1
        prediction = item.get("prediction", "").strip()
        expected = item.get("agreement", "").strip()

        if not (
            prediction.startswith("Evet")
            or prediction.startswith("Hayır")
            or "Evet" in prediction
            or "Hayır" in prediction
        ):
            print(
                f"Warning: In file {file_path}, claim_id {item.get('claim_id')} "
                f"prediction does not contain 'Evet' or 'Hayır'."
            )
        else:
            first_token = prediction.split(".")[0]
            if "Evet" in first_token:
                first_token = "Evet"
            elif "Hayır" in first_token:
                first_token = "Hayır"
            else:
                first_token = "Unknown"

            if first_token == expected:
                agreements += 1

    return agreements / total_claims if total_claims > 0 else 0.0


def main(
    pred_dir,
    responding_list,
    actual_list,
    columnist_to_code,
    columnist_to_name,
    columnist_to_newspaper,
    plot_title="Columnist Agreement Rate Heatmap",
    x_label="Actual Columnist (Claims Owner)",
    y_label="Responding Columnist LLM",
    show_columnist_legend=False,
    show_color_legend=False,
    highlight_threshold=0.0,
    use_full_range=False,
    cmap="viridis",
    use_discrete_colors=False,
    num_discrete_colors=5,
):
    """
    Plots a heatmap with:
    - X-axis: Full name + newspaper on separate lines
    - Y-axis: Abbreviated codes
    - Optional legends:
        * Columnist legend (show_columnist_legend)
        * Color legend (show_color_legend)
    - highlight_threshold (float): highlight cells with agreement rate ≤ threshold
    - use_full_range (bool): color range is [matrix.min, matrix.max] if True, else [0,1]
    - use_discrete_colors (bool): divides the color scale into discrete steps if True
    - num_discrete_colors (int): number of discrete steps
    - cmap (str): name of the colormap

    Cells with an agreement rate ≤ highlight_threshold are outlined in black.
    """
    # -------------------------------------------------------------------------
    # 1) Build the agreement matrix
    # -------------------------------------------------------------------------
    matrix = np.zeros((len(responding_list), len(actual_list)))
    for i, resp in enumerate(responding_list):
        for j, actual in enumerate(actual_list):
            file_name = f"{resp}_to_{actual}.json"
            file_path = os.path.join(pred_dir, file_name)
            if os.path.exists(file_path):
                rate = calculate_agreement_rate(file_path)
                matrix[i, j] = rate
            else:
                print(f"File not found: {file_path}")
                matrix[i, j] = 1  # or np.nan, or some default

    print("Agreement Rate Matrix:")
    print(matrix)

    # -------------------------------------------------------------------------
    # 2) Determine color range (vmin, vmax)
    # -------------------------------------------------------------------------
    if use_full_range:
        valid_vals = matrix[~np.isnan(matrix)]
        vmin = valid_vals.min() if len(valid_vals) > 0 else 0.0
        vmax = valid_vals.max() if len(valid_vals) > 0 else 1.0
    else:
        vmin = 0.0
        vmax = 1.0

    # -------------------------------------------------------------------------
    # 3) Prepare discrete or continuous colormap & normalization
    # -------------------------------------------------------------------------
    if use_discrete_colors:
        # Create a discrete colormap with `num_discrete_colors` steps from the chosen `cmap`
        colors = sns.color_palette(cmap, num_discrete_colors)
        discrete_cmap = mpl.colors.ListedColormap(colors)

        boundaries = np.linspace(vmin, vmax, num_discrete_colors + 1)
        boundaries = [round(b, 2) for b in boundaries]  # round for nicer display
        norm = mpl.colors.BoundaryNorm(boundaries, discrete_cmap.N)
        final_cmap = discrete_cmap
    else:
        # Use a continuous colormap
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        final_cmap = mpl.cm.get_cmap(cmap)

    # -------------------------------------------------------------------------
    # 4) Prepare labels
    # -------------------------------------------------------------------------
    xticklabels = [
        f"{columnist_to_name.get(act, act)}\n({columnist_to_newspaper.get(act, 'Unknown')})"
        for act in actual_list
    ]
    yticklabels = [columnist_to_code.get(resp, resp) for resp in responding_list]

    # -------------------------------------------------------------------------
    # 5) Plot the heatmap
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.10, right=0.9)  # space for legend & colorbar if needed

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=final_cmap,
        norm=norm,
        cbar=False,  # We'll add our own colorbar if needed
        ax=ax,
    )

    # Make the X-axis labels horizontal
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va="top")
    for label in ax.get_xticklabels():
        if "İsmail Saymaz" in label.get_text():
            label.set_y(label.get_position()[1] + 0.0035)  # shift up slightly

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)

    # Highlight cells with an agreement rate ≤ highlight_threshold
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            if value <= highlight_threshold:
                ax.add_patch(
                    patches.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2)
                )

    # -------------------------------------------------------------------------
    # 6) Optional Columnist Legend on the top-right
    # -------------------------------------------------------------------------
    if show_columnist_legend:
        all_columnists = sorted(set(responding_list + actual_list))
        legend_handles = []
        for col_key in all_columnists:
            code = columnist_to_code.get(col_key, col_key)
            full_name = columnist_to_name.get(col_key, col_key)
            handle = Line2D(
                [0],
                [0],
                marker="s",
                color="white",
                label=f"{code} = {full_name}",
                markerfacecolor="gray",
                markersize=10,
            )
            legend_handles.append(handle)

        legend = ax.legend(
            handles=legend_handles,
            title="Columnist Mapping",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )
        ax.add_artist(legend)

    # -------------------------------------------------------------------------
    # 7) Optional small horizontal colorbar
    # -------------------------------------------------------------------------
    if show_color_legend:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=final_cmap)
        sm.set_array([])

        # Adjust the position if the columnist legend is or isn't shown
        cbar_x0 = 0.82
        cbar_y0 = 0.60
        cbar_width = 0.15
        cbar_height = 0.02
        cbar_ax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_height])

        if use_discrete_colors:
            boundaries = np.linspace(vmin, vmax, num_discrete_colors + 1)
            boundaries = [round(b, 2) for b in boundaries]
            ticks = boundaries
            cbar = fig.colorbar(
                sm,
                cax=cbar_ax,
                orientation="horizontal",
                boundaries=boundaries,
                ticks=ticks,
            )
        else:
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

        cbar.set_label("Agreement Rate")

    plt.savefig("siu_plots/heatmap_tr.pdf", format="pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # Directory where prediction JSON files are stored
    pred_dir = "./siu_predictions"

    # Example usage
    responding_list = [
        "hilalkaplan",
        "abdulkadirselvi",
        "ahmethakan",
        "nagehanalci",
        "melihasik",
        "ismailsaymaz",
        "mehmettezkan",
        "fehimtastekin",
        "basemodel",
    ]

    actual_list = [
        "hilalkaplan",
        "abdulkadirselvi",
        "ahmethakan",
        "nagehanalci",
        "melihasik",
        "ismailsaymaz",
        "mehmettezkan",
        "fehimtastekin",
    ]

    columnist_to_code = {
        "hilalkaplan": "H.K.",
        "ahmethakan": "A.H.",
        "abdulkadirselvi": "A.S.",
        "ismailsaymaz": "İ.S.",
        "mehmettezkan": "M.T.",
        "nagehanalci": "N.A.",
        "fehimtastekin": "F.T.",
        "melihasik": "M.A.",
        "basemodel": "B.M.",
    }

    columnist_to_name = {
        "hilalkaplan": "Hilal Kaplan",
        "ahmethakan": "Ahmet Hakan",
        "abdulkadirselvi": "Abdulkadir Selvi",
        "ismailsaymaz": "İsmail Saymaz",
        "mehmettezkan": "Mehmet Tezkan",
        "nagehanalci": "Nagehan Alçı",
        "fehimtastekin": "Fehim Taştekin",
        "melihasik": "Melih Aşık",
        "basemodel": "Base Model",
    }

    columnist_to_newspaper = {
        "hilalkaplan": "Sabah",
        "ahmethakan": "Hürriyet",
        "abdulkadirselvi": "Hürriyet",
        "ismailsaymaz": "Halk TV",
        "mehmettezkan": "Halk TV",
        "nagehanalci": "Habertürk",
        "fehimtastekin": "BBC Türkçe",
        "melihasik": "Milliyet",
        "basemodel": "Gemini",
    }

    show_columnist_legend = False
    show_color_legend = False
    highlight_threshold = 0.0
    use_full_range = True
    cmap = "viridis"
    use_discrete_colors = True
    num_discrete_colors = 6

    plot_title = "Köşe Yazarları Katılım Oranı Isı Haritası"
    x_label = "Gerçek Köşe Yazarı (İddia Sahibi)"
    y_label = "Yanıtlayan Köşe Yazarı Modeli"

    main(
        pred_dir,
        responding_list,
        actual_list,
        columnist_to_code,
        columnist_to_name,
        columnist_to_newspaper,
        plot_title=plot_title,
        x_label=x_label,
        y_label=y_label,
        show_columnist_legend=show_columnist_legend,
        show_color_legend=show_color_legend,
        highlight_threshold=highlight_threshold,
        use_full_range=use_full_range,
        cmap=cmap,
        use_discrete_colors=use_discrete_colors,
        num_discrete_colors=num_discrete_colors,
    )
