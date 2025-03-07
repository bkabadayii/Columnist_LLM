import math
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

# --------------------------------------------------------------------
# 1) Columnist Setup
# --------------------------------------------------------------------
COLUMNIST_DISPLAY_NAMES = {
    "hilalkaplan": "Hilal Kaplan",
    "abdulkadirselvi": "Abdulkadir Selvi",
    "ahmethakan": "Ahmet Hakan",
    "melihasik": "Melih Aşık",
    "nagehanalci": "Nagehan Alçı",
    "ismailsaymaz": "İsmail Saymaz",
    "mehmettezkan": "Mehmet Tezkan",
    "fehimtastekin": "Fehim Taştekin",
}
COLUMNIST_IDS = list(COLUMNIST_DISPLAY_NAMES.keys())


def get_columnist_color(col_id):
    """
    Assign each columnist a color code (C0, C1, etc.) based on their index.
    """
    idx = COLUMNIST_IDS.index(col_id)
    return f"C{idx % 10}"  # cycles through up to 10 colors


# --------------------------------------------------------------------
# 2) Helper Functions
# --------------------------------------------------------------------
def load_rating_data(columnist_id):
    file_path = (
        f"../../../siu_questionnaire/agreement_ratings/{columnist_id}_ratings.json"
    )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None


def calculate_vector(result_data, question_filter=None):
    """
    For each question in question_filter (or all questions if None/empty),
    compute the average 'agreement_score'. This produces a vector.
    """
    if not result_data:
        return []

    if not question_filter:
        question_ids = sorted(result_data.keys())
    else:
        question_ids = question_filter

    vector = []
    for q_id in question_ids:
        q_entry = result_data.get(q_id)
        if not q_entry:
            vector.append(0.0)
            continue
        responses = q_entry.get("responses", [])
        scores = [
            r.get("agreement_score", 0.0)
            for r in responses
            if r.get("agreement_score") is not None
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        vector.append(avg_score)
    return vector


def remove_borders(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


# --------------------------------------------------------------------
# 3) PCA Functions
# --------------------------------------------------------------------
def perform_1d_pca(model_vectors):
    """
    Standard-scales model_vectors and performs PCA (1 component).
    Returns the flattened principal component scores.
    """
    X = np.array(model_vectors)
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros(X.shape[0])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced.flatten()


def perform_2d_pca(model_vectors):
    """
    Standard-scales model_vectors and performs PCA (2 components).
    Returns:
        scores: The projected data (shape: [n_models, 2])
        loadings: The PCA loadings (shape: [2, n_features])
    """
    X = np.array(model_vectors)
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0], 2)), None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_  # shape: (2, n_features)
    return scores, loadings


# --------------------------------------------------------------------
# 4) Plotting Functions
# --------------------------------------------------------------------
def plot_1d_pca(ax, pc_scores, columnist_ids, title):
    """
    Plots a 1D PCA result on the given axis, removing borders,
    drawing a thick horizontal line at y=0, with symmetric numeric ticks.
    """
    remove_borders(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    if len(pc_scores) > 0:
        min_x, max_x = np.min(pc_scores), np.max(pc_scores)
        max_abs = max(abs(min_x), abs(max_x))
        if max_abs == 0:
            max_abs = 1
    else:
        max_abs = 1

    margin = max_abs * 0.1
    x_left = -(max_abs + margin)
    x_right = max_abs + margin

    ax.axhline(0, color="black", linewidth=2, zorder=1)

    n_ticks = 5
    tick_positions = np.linspace(-max_abs, max_abs, n_ticks)
    for t in tick_positions:
        ax.plot([t, t], [0, 0], marker="|", markersize=10, color="black", zorder=2)
        ax.text(t, -0.07, f"{t:.1f}", ha="center", va="top", fontsize=8, zorder=3)

    colors = [get_columnist_color(cid) for cid in columnist_ids]
    ax.scatter(pc_scores, np.zeros_like(pc_scores), s=100, c=colors, zorder=4)
    for x_val, col_id in zip(pc_scores, columnist_ids):
        display_name = COLUMNIST_DISPLAY_NAMES.get(col_id, col_id)
        ax.text(
            x_val,
            0.05,
            display_name,
            rotation=45,
            va="bottom",
            ha="left",
            fontsize=9,
            zorder=5,
        )

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(-0.3, 0.7)
    ax.set_title(title, fontsize=12)


def plot_biplot(ax, scores, loadings, feature_names, columnist_ids, title):
    """
    Plots a biplot on the given axis.
    - 'scores' is the 2D projection of the data (n_models x 2).
    - 'loadings' is a (2, n_features) array representing the PCA loadings.
    - 'feature_names' is a list of names for each feature (columns).

    The biplot shows:
      - A scores scatter plot (each point colored by columnist).
      - Loadings as arrows from the origin.
    """
    remove_borders(ax)
    ax.set_xticks([])
    ax.set_yticks([])

    # Determine symmetric limits based on scores
    if scores.shape[0] > 0:
        x_min, x_max = np.min(scores[:, 0]), np.max(scores[:, 0])
        y_min, y_max = np.min(scores[:, 1]), np.max(scores[:, 1])
        lim_x = max(abs(x_min), abs(x_max)) * 1.2
        lim_y = max(abs(y_min), abs(y_max)) * 1.2
    else:
        lim_x, lim_y = 1, 1

    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)

    # Draw horizontal and vertical lines at 0
    ax.axhline(0, color="black", linewidth=1, zorder=1)
    ax.axvline(0, color="black", linewidth=1, zorder=1)

    # Plot the scores with distinct colors
    colors = [get_columnist_color(cid) for cid in columnist_ids]
    ax.scatter(scores[:, 0], scores[:, 1], s=100, c=colors, zorder=3)
    for (x, y), col_id in zip(scores, columnist_ids):
        display_name = COLUMNIST_DISPLAY_NAMES.get(col_id, col_id)
        ax.text(
            x,
            y,
            display_name,
            fontsize=9,
            rotation=45,
            va="bottom",
            ha="left",
            zorder=4,
        )

    # Scale loadings for visibility (scale factor can be adjusted)
    # Here we use a fraction of lim_x (or lim_y)
    loading_scale = lim_x * 0.8
    n_features = loadings.shape[1]
    for j in range(n_features):
        vec = loadings[:, j] * loading_scale
        ax.arrow(
            0, 0, vec[0], vec[1], color="red", width=0.005, head_width=0.05, zorder=2
        )
        if feature_names and j < len(feature_names):
            ax.text(
                vec[0] * 1.1,
                vec[1] * 1.1,
                feature_names[j],
                color="red",
                fontsize=8,
                zorder=5,
            )

    ax.set_title(title, fontsize=12)


# --------------------------------------------------------------------
# 6) Reusable PCA Plot Function
# --------------------------------------------------------------------
def plot_multiple_pca(pca_configs, save_path=None, use_biplot=False):
    """
    Takes a list of (title, question_filter) pairs (pca_configs) and creates a figure
    with one subplot per pair in a 2-column layout. Standard scaling is applied before PCA.

    If use_biplot is True, performs 2D PCA and produces a biplot.
    Otherwise, performs 1D PCA and produces the standard 1D plot.
    """
    n_plots = len(pca_configs)
    if n_plots == 0:
        print("No PCA configs provided.")
        return

    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        axes = [ax]
    else:
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

    for i, (title, q_filter) in enumerate(pca_configs):
        ax = axes[i]
        model_vectors = []
        valid_columnists = []
        feature_names = None

        # Build vectors for each columnist
        for col_id in COLUMNIST_IDS:
            data = load_rating_data(col_id)
            if data is None:
                continue
            if not q_filter and feature_names is None:
                feature_names = sorted(data.keys())
            vec = calculate_vector(data, question_filter=q_filter)
            model_vectors.append(vec)
            valid_columnists.append(col_id)

        if use_biplot:
            # Perform 2D PCA
            scores, loadings = perform_2d_pca(model_vectors)
            # Use feature_names if available; if not, use q_filter.
            if feature_names is None:
                feature_names = q_filter
            plot_biplot(ax, scores, loadings, feature_names, valid_columnists, title)
        else:
            # Perform 1D PCA
            pc_scores = perform_1d_pca(model_vectors)
            plot_1d_pca(ax, pc_scores, valid_columnists, title)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(
        top=0.90, bottom=0.08, left=0.07, right=0.9, hspace=0.4, wspace=0.4
    )
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()


def perform_2d_pca(model_vectors):
    """
    Standard-scales model_vectors and performs PCA with 2 components.
    Returns:
        scores: The projected data (n_models x 2).
        loadings: The PCA loadings (2 x n_features).
    """
    X = np.array(model_vectors)
    if X.size == 0 or X.shape[1] == 0:
        return np.zeros((X.shape[0], 2)), None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_
    return scores, loadings


# --------------------------------------------------------------------
# 5) PCA Feature Importance Heatmap Function
# --------------------------------------------------------------------
def plot_pca_importance_heatmap(question_filter=None, save_path=None):
    """
    Computes the PCA loadings for all features (using standard scaling) across all
    columnist rating files, then reorganizes these loadings into a heatmap.

    The heatmap has rows = base questions: ["q1", "q2", "q3", "q4", "q5", "q6"]
    and columns = parties: ["SELF", "AKP", "CHP", "HDP", "MHP", "IYI"].

    A feature key with an underscore (e.g. "akp_q3") is interpreted as:
      - Party: uppercase part before the underscore ("AKP")
      - Base question: part after the underscore ("q3")
    Features without an underscore belong to SELF.
    The cell value is the absolute loading (importance) from the first principal component.
    """
    # Gather model vectors from all columnists
    all_vectors = []
    # For feature names, use the keys from the first valid file if no filter provided.
    feature_names = None
    for col_id in COLUMNIST_IDS:
        data = load_rating_data(col_id)
        if data is None:
            continue
        if (
            question_filter is None or len(question_filter) == 0
        ) and feature_names is None:
            feature_names = sorted(data.keys())
        vec = calculate_vector(data, question_filter=question_filter)
        all_vectors.append(vec)
    if feature_names is None:
        # If filter was provided, then use it.
        feature_names = question_filter

    X = np.array(all_vectors)  # shape: (n_columnists, n_features)
    if X.size == 0 or X.shape[1] == 0:
        print("No data available for PCA importance heatmap.")
        return

    # Standard scale X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    loadings = pca.components_[0]  # First principal component loadings

    # Create heatmap matrix: rows are base questions, columns are parties
    base_questions = ["q1", "q2", "q3", "q4", "q5", "q6"]
    parties = ["SELF", "AKP", "CHP", "HDP", "MHP", "IYI"]
    importance_matrix = np.zeros((len(base_questions), len(parties)))

    # For each feature, parse the name and populate the heatmap matrix
    for fname, loading in zip(feature_names, loadings):
        if "_" in fname:
            parts = fname.split("_", 1)
            party = parts[0].upper()
            base = parts[1]
        else:
            party = "SELF"
            base = fname
        if base in base_questions and party in parties:
            row_idx = base_questions.index(base)
            col_idx = parties.index(party)
            importance_matrix[row_idx, col_idx] = np.abs(loading)

    # Create discrete colormap: split "viridis" into 6 equal parts
    n_discrete = 6
    colors = sns.color_palette("viridis", n_discrete)

    discrete_cmap = ListedColormap(colors)
    bounds = np.linspace(
        importance_matrix.min(), importance_matrix.max(), n_discrete + 1
    )
    norm = BoundaryNorm(bounds, n_discrete)

    # Plot the heatmap without colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        importance_matrix,
        annot=True,
        cmap=discrete_cmap,
        norm=norm,
        cbar=False,
        xticklabels=parties,
        yticklabels=base_questions,
        square=True,
        ax=ax,
    )
    # Rotate y-axis labels 90 degrees
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("PCA Feature Importance Heatmap", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()


# --------------------------------------------------------------------
# 7) Cosine Similarity Heatmap Function (unchanged)
# --------------------------------------------------------------------
def plot_cosine_similarity_heatmap(
    question_filter=None, plot_title="Cosine Similarity", save_path=None
):
    vectors = []
    col_ids = []
    for col_id in COLUMNIST_IDS:
        data = load_rating_data(col_id)
        if data is None:
            continue
        vec = calculate_vector(data, question_filter=question_filter)
        vectors.append(vec)
        col_ids.append(col_id)
    X = np.array(vectors)
    similarity_matrix = cosine_similarity(X, X)
    fig, ax = plt.subplots(figsize=(8, 6))
    row_labels = [COLUMNIST_DISPLAY_NAMES[cid] for cid in col_ids]
    col_labels = [COLUMNIST_DISPLAY_NAMES[cid] for cid in col_ids]
    sns.heatmap(
        similarity_matrix,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="viridis",
        annot=True,
        vmin=0.0,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(plot_title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()


# --------------------------------------------------------------------
# 8) Main Function
# --------------------------------------------------------------------
def main():
    pca_configs_1 = [("ALL", [])]
    pca_configs_2 = [
        ("SELF", ["q1", "q2", "q3", "q4", "q5", "q6"]),
        ("AKP", ["akp_q1", "akp_q2", "akp_q3", "akp_q4", "akp_q5", "akp_q6"]),
        ("CHP", ["chp_q1", "chp_q2", "chp_q3", "chp_q4", "chp_q5", "chp_q6"]),
        ("HDP", ["hdp_q1", "hdp_q2", "hdp_q3", "hdp_q4", "hdp_q5", "hdp_q6"]),
        ("MHP", ["mhp_q1", "mhp_q2", "mhp_q3", "mhp_q4", "mhp_q5", "mhp_q6"]),
        ("IYI", ["iyi_q1", "iyi_q2", "iyi_q3", "iyi_q4", "iyi_q5", "iyi_q6"]),
    ]
    pca_configs_3 = [
        ("SELF", ["q1", "q2", "q3", "q4", "q5", "q6"]),
    ]

    # Example: Plot 1D PCA plots (default use_biplot=False)
    print("Plotting PCA (ALL questions) ...")
    plot_multiple_pca(pca_configs_1, save_path="../../../siu_plots/pca_all.pdf")

    print("Plotting PCA (party questions) as 1D plots ...")
    plot_multiple_pca(pca_configs_2, save_path="../../../siu_plots/pca_party.pdf")

    # Example: Plot biplots (2D PCA)
    print("Plotting PCA as Biplots ...")
    plot_multiple_pca(
        pca_configs_3,
        save_path="../../../siu_plots/pca_self_biplot.pdf",
        use_biplot=True,
    )

    print("Plotting cosine similarity heatmap (ALL questions) ...")
    plot_cosine_similarity_heatmap(
        question_filter=[],
        plot_title="Cosine Similarity",
        save_path="../../../siu_plots/cosine_all.pdf",
    )

    print("Plotting cosine similarity heatmap (SELF questions) ...")
    plot_cosine_similarity_heatmap(
        question_filter=["q1", "q2", "q3", "q4", "q5", "q6"],
        plot_title="Cosine Similarity Self Questions",
        save_path="../../../siu_plots/cosine_self.pdf",
    )

    plot_pca_importance_heatmap(
        question_filter=[], save_path="../../../siu_plots/pca_importance.pdf"
    )


if __name__ == "__main__":
    main()
