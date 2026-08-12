"""
linguistic_metrics.py — Article-level linguistic feature extraction.

Reads:  data/processed/articles_deduplicated.csv
Writes: reports/tables/linguistic_metrics.csv

Metrics computed per article:
  - word_count, sentence_count, char_count
  - avg_sentence_length, median_sentence_length
  - unique_word_count
  - ttr           (Type-Token Ratio — length-sensitive)
  - root_ttr      (Root TTR = unique / sqrt(total))
  - mattr         (Moving-Average TTR, window=100 — length-robust)
  - hapax_ratio   (once-occurring words / total unique words)
  - func_word_ratio
  - pos_* distributions (noun/verb/adj/adv fractions)
  - readability scores (Flesch, FK Grade, Dale-Chall, SMOG, ARI)

Run from the project root:
  python -m src.linguistic_metrics
"""

import sys
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import textstat

from src.config import (
    DEDUPED_CSV,
    TABLES_DIR,
    SPACY_MODEL,
    MATTR_WINDOW,
    READABILITY_METRICS,
)
from src.utils import get_logger, safe_div

logger = get_logger(__name__)

# ─── spaCy lazy load ──────────────────────────────────────────────────────────

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
            _nlp.add_pipe("sentencizer")
            logger.info("Loaded spaCy model: %s", SPACY_MODEL)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                SPACY_MODEL, SPACY_MODEL,
            )
            sys.exit(1)
    return _nlp


# ─── English function words ───────────────────────────────────────────────────

FUNCTION_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","up","about","into","through","during","including","until",
    "against","among","throughout","despite","towards","upon","concerning",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","need",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","his","its","our","their","this","that","these","those",
    "who","which","what","where","when","how","why","whom",
    "not","no","nor","so","yet","both","either","neither","each","few",
    "more","most","other","some","such","than","too","very","just","as",
    "if","then","because","while","although","though","unless","since",
}


# ─── Core metric functions ────────────────────────────────────────────────────

def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    """
    Moving-Average Type-Token Ratio.
    More length-robust than simple TTR.
    """
    if len(tokens) < window:
        unique = len(set(tokens))
        return safe_div(unique, len(tokens))

    ttrs = []
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i: i + window]
        ttrs.append(safe_div(len(set(window_tokens)), window))
    return float(np.mean(ttrs))


def hapax_ratio(tokens: list[str]) -> float:
    """Fraction of unique word types that appear exactly once."""
    freq = Counter(tokens)
    n_hapax  = sum(1 for v in freq.values() if v == 1)
    n_types  = len(freq)
    return safe_div(n_hapax, n_types)


def func_word_ratio(tokens: list[str]) -> float:
    """Fraction of tokens that are function words."""
    n_func = sum(1 for t in tokens if t.lower() in FUNCTION_WORDS)
    return safe_div(n_func, len(tokens))


def pos_distribution(doc) -> dict:
    """Return fractional POS distribution for a spaCy Doc."""
    total = len([t for t in doc if not t.is_space])
    if total == 0:
        return {k: 0.0 for k in ["pos_noun","pos_verb","pos_adj","pos_adv","pos_propn"]}
    counts = Counter(t.pos_ for t in doc if not t.is_space)
    return {
        "pos_noun":  safe_div(counts.get("NOUN",  0), total),
        "pos_verb":  safe_div(counts.get("VERB",  0), total),
        "pos_adj":   safe_div(counts.get("ADJ",   0), total),
        "pos_adv":   safe_div(counts.get("ADV",   0), total),
        "pos_propn": safe_div(counts.get("PROPN", 0), total),
    }


def readability_scores(text: str) -> dict:
    """Calculate readability metrics using textstat."""
    scores = {}
    for metric in READABILITY_METRICS:
        try:
            scores[metric] = getattr(textstat, metric)(text)
        except Exception:
            scores[metric] = None
    return scores


def compute_metrics(text: str) -> dict:
    """
    Compute all linguistic metrics for a single article text.
    Returns a flat dict of metric name → value.
    """
    if not text or len(text.strip()) < 10:
        return {}

    nlp = get_nlp()

    # spaCy truncates at 1M chars by default — safe for articles
    doc = nlp(text[:500_000])

    # Tokens (lowercase, alpha-only for most metrics)
    all_tokens  = [t.text for t in doc if not t.is_space]
    # NOTE: spaCy Token.lower is an int hash ID; Token.lower_ is the string form.
    alpha_tokens = [t.lower_ for t in doc if t.is_alpha and not t.is_space]

    # Sentences
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in s if not t.is_space]) for s in sentences]

    # Vocabulary metrics
    n_tokens  = len(alpha_tokens)
    n_unique  = len(set(alpha_tokens))
    ttr_val   = safe_div(n_unique, n_tokens)
    root_ttr  = safe_div(n_unique, n_tokens ** 0.5) if n_tokens else 0.0
    mattr_val = mattr(alpha_tokens)
    hapax     = hapax_ratio(alpha_tokens)
    fw_ratio  = func_word_ratio(alpha_tokens)

    # Sentence structure
    n_sents   = len(sentences)
    avg_sl    = float(np.mean(sent_lengths))  if sent_lengths else 0.0
    med_sl    = float(np.median(sent_lengths)) if sent_lengths else 0.0

    result = {
        "token_count":       n_tokens,
        "sentence_count":    n_sents,
        "avg_sentence_len":  round(avg_sl, 2),
        "med_sentence_len":  round(med_sl, 2),
        "unique_word_count": n_unique,
        "ttr":               round(ttr_val, 4),
        "root_ttr":          round(root_ttr, 4),
        "mattr":             round(mattr_val, 4),
        "hapax_ratio":       round(hapax, 4),
        "func_word_ratio":   round(fw_ratio, 4),
    }
    result.update(pos_distribution(doc))
    result.update(readability_scores(text))
    return result


# ─── Batch processor ─────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame, n_sample: Optional[int] = None) -> pd.DataFrame:
    """
    Compute linguistic metrics for every row in df.
    Optionally sample n_sample rows for speed during development.
    Returns a DataFrame with article_id + all metric columns.
    """
    if n_sample:
        df = df.sample(min(n_sample, len(df)), random_state=42)

    logger.info("Computing metrics for %d articles …", len(df))
    rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 500 == 0:
            logger.info("  %d / %d", i, len(df))
        metrics = compute_metrics(str(row.get("text", "")))
        metrics["article_id"] = row["article_id"]
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    return metrics_df


# ─── Yearly aggregation ───────────────────────────────────────────────────────

def yearly_metrics(df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Merge article metadata with metrics and compute yearly medians."""
    merged = df[["article_id","year","corpus","article_type"]].merge(
        metrics_df, on="article_id", how="left"
    )

    numeric_cols = metrics_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "article_id"]

    agg = {c: ["median","mean","std"] for c in numeric_cols}
    yearly = merged.groupby("year").agg(agg).reset_index()
    yearly.columns = ["_".join(c).strip("_") for c in yearly.columns]
    return yearly


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_linguistic_metrics() -> pd.DataFrame:
    if not DEDUPED_CSV.exists():
        logger.error("No deduplicated CSV found. Run quality.py first.")
        sys.exit(1)

    df = pd.read_csv(DEDUPED_CSV, low_memory=False)
    logger.info("Loaded %d articles", len(df))

    metrics_df = compute_all_metrics(df)

    # Save article-level metrics
    out = df[["article_id","year","month","quarter","corpus","article_type","word_count"]].merge(
        metrics_df, on="article_id", how="left"
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "linguistic_metrics.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved article-level metrics → %s", out_path)

    # Save yearly summary
    ym = yearly_metrics(df, metrics_df)
    ym_path = TABLES_DIR / "linguistic_metrics_by_year.csv"
    ym.to_csv(ym_path, index=False)
    logger.info("Saved yearly metrics → %s", ym_path)

    return out


if __name__ == "__main__":
    run_linguistic_metrics()
