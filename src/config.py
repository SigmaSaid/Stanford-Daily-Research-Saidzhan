"""
config.py — Centralized configuration for the Stanford Daily research project.

All parameters are defined here. Do NOT hard-code these values in other modules.
Edit this file to change scraping behaviour, analysis settings, or output paths.
"""

import os
from pathlib import Path
from datetime import datetime

# ─── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR    / "raw"
PROCESSED_DIR   = DATA_DIR    / "processed"
EMBEDDINGS_DIR  = DATA_DIR    / "embeddings"
REPORTS_DIR     = PROJECT_ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
TABLES_DIR      = REPORTS_DIR / "tables"
SRC_DIR         = PROJECT_ROOT / "src"
LOGS_DIR        = PROJECT_ROOT / "logs"

# Ensure output directories exist at import time
for _d in [RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, REPORTS_DIR, FIGURES_DIR,
           TABLES_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Pipeline log (project-relative, not cwd-relative)
PIPELINE_LOG = LOGS_DIR / "pipeline.log"

# ─── Collection parameters ────────────────────────────────────────────────────
BASE_URL       = "https://stanforddaily.com"
API_BASE       = f"{BASE_URL}/wp-json/wp/v2"
POSTS_ENDPOINT = f"{API_BASE}/posts"
CATS_ENDPOINT  = f"{API_BASE}/categories"
TAGS_ENDPOINT  = f"{API_BASE}/tags"

# Date window
START_DATE = "2015-01-01"
END_DATE   = "2026-08-10"

# Target categories — names to search for (IDs discovered at runtime)
OPINION_CATEGORY_NAMES = ["opinions", "opinion", "op-ed", "op-eds", "letters",
                          "columnists", "editorial", "editorials"]
NEWS_CATEGORY_NAMES    = ["news", "university", "local", "national", "world"]

# Pagination
API_PER_PAGE = 100          # WordPress max = 100
MAX_API_PAGES = 2000        # Safety ceiling

# Rate limiting
REQUEST_DELAY_SECONDS     = 0.5   # Seconds between requests
REQUEST_DELAY_JITTER      = 0.25  # ± random jitter added to delay
REQUEST_TIMEOUT_SECONDS   = 30    # Per-request timeout
MAX_RETRIES               = 5     # Retry attempts per request
RETRY_BACKOFF_FACTOR      = 2.0   # Exponential backoff multiplier

# User-agent
USER_AGENT = (
    "StanfordDailyLinguisticResearch/1.0 "
    "(academic study; contact: researcher@example.edu)"
)

# Output files
RAW_JSONL        = RAW_DIR / "articles_raw.jsonl"
COLLECTION_LOG   = RAW_DIR / "collection.log"
FAILED_LOG       = RAW_DIR / "failed_articles.log"
CATEGORY_CACHE   = RAW_DIR / "categories.json"
AUTHOR_CACHE     = RAW_DIR / "authors.json"

# ─── Processing parameters ────────────────────────────────────────────────────
PROCESSED_CSV       = PROCESSED_DIR / "articles.csv"
DEDUPED_CSV         = PROCESSED_DIR / "articles_deduplicated.csv"
DEDUP_REMOVED_CSV   = PROCESSED_DIR / "dedup_removed.csv"

# Article length filters
MIN_WORD_COUNT = 50    # Shorter articles likely nav/stub pages
MAX_WORD_COUNT = 15000 # Extremely long articles may be aggregations

# ─── Linguistic analysis ──────────────────────────────────────────────────────
SPACY_MODEL    = "en_core_web_sm"
MATTR_WINDOW   = 100   # Moving Average TTR window size

# Readability — which scores to compute
READABILITY_METRICS = ["flesch_reading_ease", "flesch_kincaid_grade",
                       "dale_chall_readability_score", "smog_index",
                       "automated_readability_index"]

# ─── Vocabulary & n-grams ─────────────────────────────────────────────────────
NGRAM_SIZES    = [2, 3]
TOP_N_WORDS    = 200   # Top words per year for TF-IDF
TOP_N_NGRAMS   = 100   # Top n-grams per year

# Exploratory "AI-associated" vocabulary list (NOT a validated detector)
# Source: recurring discourse in public media 2023-2025 about LLM writing style.
# This list is exploratory only; treat findings as hypothesis-generating, not confirmatory.
AI_EXPLORATORY_VOCAB = [
    "delve", "delves", "delved", "delving",
    "underscore", "underscores", "underscored", "underscoring",
    "nuanced", "nuance", "nuances",
    "multifaceted",
    "foster", "fosters", "fostered", "fostering",
    "landscape", "landscapes",
    "tapestry", "tapestries",
    "crucial", "crucially",
    "arguably",
    "notably",
    "paramount",
    "intricate", "intricately",
    "comprehensive", "comprehensively",
    "robust", "robustly",
    "leverage", "leverages", "leveraged", "leveraging",
    "harness", "harnessing", "harnessed",
    "pivotal",
    "realm", "realms",
    "vibrant",
    "dynamic", "dynamics",
    "holistic", "holistically",
    "groundbreaking",
    "innovative", "innovation",
    "transformative",
    "unprecedented",
    "synergy", "synergies", "synergistic",
    "streamline", "streamlines", "streamlined",
    "ecosystem", "ecosystems",
]

# Periods for aggregated comparisons
PERIODS = {
    "pre_ai":       ("2015-01-01", "2019-12-31"),
    "transition":   ("2020-01-01", "2022-12-31"),
    "chatgpt_era":  ("2023-01-01", "2026-08-10"),
}

# Fine-grained annual bins for change-point analysis
YEARS_OF_INTEREST = list(range(2015, 2027))

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_NPY       = EMBEDDINGS_DIR / "embeddings.npy"
EMBEDDING_IDS_CSV    = EMBEDDINGS_DIR / "article_ids.csv"
EMBEDDING_BATCH_SIZE = 64

# Maximum articles to embed (use None for all; reduce if memory is limited)
MAX_EMBED_ARTICLES = None

# ─── Similarity analysis ──────────────────────────────────────────────────────
SIMILARITY_SAMPLE_PER_YEAR = 300   # Articles per year sampled for pairwise similarity
SIMILARITY_RANDOM_SEED     = 42

# ─── Topic modeling ───────────────────────────────────────────────────────────
NUM_TOPICS       = 15
NMF_MAX_ITER     = 500
NMF_RANDOM_STATE = 42
MAX_TFIDF_FEATURES_TOPICS = 5000

# ─── Statistical analysis ─────────────────────────────────────────────────────
ALPHA              = 0.05      # Significance threshold
RANDOM_SEED        = 42
MULTIPLE_CORRECTION = "fdr_bh"  # Benjamini-Hochberg FDR

# Change-point search window (months)
CHANGEPOINT_MIN_YEAR  = 2021
CHANGEPOINT_MAX_YEAR  = 2024

# ─── Visualisation ────────────────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"
PALETTE_YEAR  = "viridis"   # Colormap for year-coded scatter plots
STYLE         = "seaborn-v0_8-whitegrid"

# ─── Reproducibility ─────────────────────────────────────────────────────────
COLLECTION_DATE = datetime.today().strftime("%Y-%m-%d")
PYTHON_VERSION  = "3.9+"
