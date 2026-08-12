"""
visualize.py — Generate all publication-quality figures for the research report.

Reads: tables from reports/tables/
Writes: PNG figures to reports/figures/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from src.config import (
    TABLES_DIR,
    FIGURES_DIR,
    EMBEDDINGS_NPY,
    EMBEDDING_IDS_CSV,
    FIGURE_DPI,
    FIGURE_FORMAT,
    PALETTE_YEAR,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_theme(style="whitegrid", context="talk", palette="muted")
plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = FIGURE_DPI


def save_fig(name: str) -> None:
    path = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure → %s", path)


# ─── 1. Articles by year ──────────────────────────────────────────────────────

def plot_articles_by_year(stats_path: Path) -> None:
    if not stats_path.exists():
        return
    df = pd.read_csv(stats_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["year"], df["n_articles"], color="steelblue", edgecolor="black")
    ax.axvline(2022.5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="ChatGPT release")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Stanford Daily Articles Collected (2015–2026)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig("01_articles_by_year")


# ─── 2. Word count trends ─────────────────────────────────────────────────────

def plot_word_count_trends(stats_path: Path) -> None:
    if not stats_path.exists():
        return
    df = pd.read_csv(stats_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["year"], df["mean_word_count"], marker="o", linewidth=2, label="Mean")
    ax.plot(df["year"], df["median_word_count"], marker="s", linewidth=2, label="Median")
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Word Count")
    ax.set_title("Average Article Length Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("02_word_count_trends")


# ─── 3. MATTR by year ─────────────────────────────────────────────────────────

def plot_mattr_by_year(metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    yearly = df.groupby("year")["mattr"].agg(["median","mean"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly["year"], yearly["median"], marker="o", linewidth=2.5, label="Median MATTR")
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5, label="ChatGPT release")
    ax.set_xlabel("Year")
    ax.set_ylabel("MATTR (Moving-Average TTR)")
    ax.set_title("Vocabulary Diversity Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("03_mattr_by_year")


# ─── 4. MATTR distribution violin ─────────────────────────────────────────────

def plot_mattr_distribution(metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    df["period"] = df["year"].apply(lambda y:
        "2015-2019" if y <= 2019 else
        "2020-2022" if y <= 2022 else
        "2023" if y == 2023 else
        "2024-2026"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="period", y="mattr", hue="period",
                   palette="Set2", legend=False, ax=ax)
    ax.set_xlabel("Period")
    ax.set_ylabel("MATTR")
    ax.set_title("Vocabulary Diversity Distribution by Period")
    ax.grid(axis="y", alpha=0.3)
    save_fig("04_mattr_distribution")


# ─── 5. Sentence length ───────────────────────────────────────────────────────

def plot_sentence_length(metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    yearly = df.groupby("year")["avg_sentence_len"].median().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly["year"], yearly["avg_sentence_len"], marker="o", linewidth=2.5, color="coral")
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median Sentence Length (words)")
    ax.set_title("Sentence Length Over Time")
    ax.grid(alpha=0.3)
    save_fig("05_sentence_length")


# ─── 6. Readability ───────────────────────────────────────────────────────────

def plot_readability(metrics_path: Path) -> None:
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)

    for metric in ["flesch_reading_ease", "flesch_kincaid_grade"]:
        if metric not in df.columns:
            continue
        yearly = df.groupby("year")[metric].median().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(yearly["year"], yearly[metric], marker="o", linewidth=2.5)
        ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Over Time")
        ax.grid(alpha=0.3)
        save_fig(f"06_{metric}")


# ─── 7. AI vocabulary trends ──────────────────────────────────────────────────

def plot_ai_vocab(ai_vocab_path: Path) -> None:
    if not ai_vocab_path.exists():
        return
    df = pd.read_csv(ai_vocab_path)

    # Aggregate across all exploratory words
    yearly_total = df.groupby("year")["freq_per_million"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_total["year"], yearly_total["freq_per_million"],
            marker="o", linewidth=3, color="purple", label="Total exploratory AI vocab")
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5, label="ChatGPT release")
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency per Million Words")
    ax.set_title("Exploratory AI-Associated Vocabulary Frequency Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("07_ai_vocab_trends")

    # Top individual words
    top_words = df.groupby("word")["freq_per_million"].mean().nlargest(8).index
    fig, ax = plt.subplots(figsize=(14, 7))
    for word in top_words:
        word_df = df[df["word"] == word]
        ax.plot(word_df["year"], word_df["freq_per_million"], marker="o", linewidth=2, label=word)
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency per Million Words")
    ax.set_title("Top Exploratory AI-Associated Words")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    save_fig("08_ai_vocab_individual")


# ─── 8. Top vocabulary changes ────────────────────────────────────────────────

def plot_vocab_changes(changes_path: Path) -> None:
    if not changes_path.exists():
        return
    df = pd.read_csv(changes_path)

    top_inc = df.head(15)
    top_dec = df.tail(15).sort_values("absolute_change")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.barh(top_inc["word"], top_inc["absolute_change"], color="green", edgecolor="black")
    ax1.set_xlabel("Frequency Change (per million)")
    ax1.set_title("Top 15 Words Increasing (2015-19 → 2024-26)")
    ax1.invert_yaxis()

    ax2.barh(top_dec["word"], top_dec["absolute_change"], color="crimson", edgecolor="black")
    ax2.set_xlabel("Frequency Change (per million)")
    ax2.set_title("Top 15 Words Decreasing")
    ax2.invert_yaxis()

    save_fig("09_vocab_changes")


# ─── 9. N-gram changes ────────────────────────────────────────────────────────

def plot_ngram_changes(ngram_changes_path: Path) -> None:
    if not ngram_changes_path.exists():
        return
    df = pd.read_csv(ngram_changes_path)

    for n in [2, 3]:
        sub = df[df["n"] == n]
        if sub.empty:
            continue
        top_inc = sub.nlargest(12, "absolute_change")

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(top_inc["ngram"], top_inc["absolute_change"], color="teal", edgecolor="black")
        ax.set_xlabel("Frequency Change (per million)")
        ax.set_title(f"Top {'Bigrams' if n==2 else 'Trigrams'} Increasing (2015-19 → 2024-26)")
        ax.invert_yaxis()
        save_fig(f"10_ngram_changes_n{n}")


# ─── 10. Semantic similarity ──────────────────────────────────────────────────

def plot_semantic_similarity(sim_path: Path) -> None:
    if not sim_path.exists():
        return
    df = pd.read_csv(sim_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["year"], df["mean_sim"], marker="o", linewidth=2.5, label="Mean within-year similarity")
    ax.fill_between(df["year"],
                     df["mean_sim"] - df["std_sim"],
                     df["mean_sim"] + df["std_sim"],
                     alpha=0.2)
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5, label="ChatGPT release")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Semantic Similarity Within Year (Article Embeddings)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig("11_semantic_similarity")


# ─── 11. PCA embeddings ───────────────────────────────────────────────────────

def plot_embedding_pca() -> None:
    emb_path = EMBEDDINGS_NPY
    ids_path = EMBEDDING_IDS_CSV

    if not emb_path.exists() or not ids_path.exists():
        logger.warning("Embeddings not found — skipping PCA plot")
        return

    emb = np.load(str(emb_path))
    ids_df = pd.read_csv(ids_path)

    logger.info("Running PCA for visualization …")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(emb)

    ids_df["pc1"] = coords[:, 0]
    ids_df["pc2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(ids_df["pc1"], ids_df["pc2"],
                         c=ids_df["year"], cmap=PALETTE_YEAR,
                         alpha=0.6, s=10, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Year")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("Article Embeddings (PCA Projection)")
    save_fig("12_embedding_pca")


# ─── 12. Topic distributions ──────────────────────────────────────────────────

def plot_topic_distributions(topic_dist_path: Path) -> None:
    if not topic_dist_path.exists():
        return
    df = pd.read_csv(topic_dist_path)

    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        return

    # Stacked area chart of top 8 topics
    mean_shares = df[topic_cols].mean().sort_values(ascending=False)
    top8 = mean_shares.head(8).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(df["year"], *[df[t] for t in top8], labels=top8, alpha=0.8)
    ax.axvline(2022.5, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic Share")
    ax.set_title("Topic Distribution Over Time (NMF, Top 8 Topics)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.2)
    save_fig("13_topic_distributions")


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating all figures …")

    # Load paths
    qc_path        = TABLES_DIR / "yearly_corpus_statistics.csv"
    metrics_path   = TABLES_DIR / "linguistic_metrics.csv"
    ai_vocab_path  = TABLES_DIR / "ai_associated_vocabulary.csv"
    vocab_chg_path = TABLES_DIR / "top_vocabulary_changes.csv"
    ngram_chg_path = TABLES_DIR / "top_ngram_changes.csv"
    sim_path       = TABLES_DIR / "semantic_similarity_stats.csv"
    topic_path     = TABLES_DIR / "topic_distributions.csv"

    # Generate
    plot_articles_by_year(qc_path)
    plot_word_count_trends(qc_path)
    plot_mattr_by_year(metrics_path)
    plot_mattr_distribution(metrics_path)
    plot_sentence_length(metrics_path)
    plot_readability(metrics_path)
    plot_ai_vocab(ai_vocab_path)
    plot_vocab_changes(vocab_chg_path)
    plot_ngram_changes(ngram_chg_path)
    plot_semantic_similarity(sim_path)
    plot_embedding_pca()
    plot_topic_distributions(topic_path)

    logger.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    generate_all_figures()
