"""
run_pipeline.py — Main execution script for the Stanford Daily research pipeline.

Runs the complete analysis pipeline from data collection through visualization.

Run from the project root:
  python -m src.run_pipeline                    # full pipeline
  python -m src.run_pipeline --skip-collection  # skip scraping (use existing data)
  python -m src.run_pipeline --quick            # fast mode: skip embeddings & topics
"""

import argparse
import sys

from src.utils import get_logger
from src.config import RAW_JSONL, DEDUPED_CSV, PIPELINE_LOG

logger = get_logger(__name__, log_file=PIPELINE_LOG)


def run_full_pipeline(skip_collection: bool = False, quick: bool = False) -> None:
    """Execute the complete research pipeline."""

    logger.info("=" * 70)
    logger.info("STANFORD DAILY AI VOCABULARY SHIFT RESEARCH PIPELINE")
    logger.info("=" * 70)

    # ─── Step 1: Metadata discovery ──────────────────────────────────────────
    if not skip_collection:
        logger.info("\n[1/12] Discovering category metadata …")
        from src.metadata import discover_targets
        discover_targets()

        # ─── Step 2: Collection ───────────────────────────────────────────────
        logger.info("\n[2/12] Collecting articles from Stanford Daily …")
        from src.collect import run_collection
        run_collection(corpus="both", reset_cache=False)

        if not RAW_JSONL.exists() or RAW_JSONL.stat().st_size == 0:
            logger.error("Collection failed or returned no data.")
            sys.exit(1)
    else:
        logger.info("\n[1-2/12] Skipping collection (--skip-collection)")
        if not RAW_JSONL.exists():
            logger.error("No raw data found and collection skipped. Cannot proceed.")
            sys.exit(1)

    # ─── Step 3: Cleaning ─────────────────────────────────────────────────────
    logger.info("\n[3/12] Cleaning HTML → text …")
    from src.clean import run_cleaning
    run_cleaning()

    # ─── Step 4: Quality checks & deduplication ───────────────────────────────
    logger.info("\n[4/12] Running quality checks and deduplication …")
    from src.quality import run_quality_pipeline
    run_quality_pipeline()

    if not DEDUPED_CSV.exists():
        logger.error("Deduplication failed. Cannot proceed.")
        sys.exit(1)

    # ─── Step 5: Linguistic metrics ───────────────────────────────────────────
    logger.info("\n[5/12] Computing linguistic metrics …")
    from src.linguistic_metrics import run_linguistic_metrics
    run_linguistic_metrics()

    # ─── Step 6: Vocabulary analysis ──────────────────────────────────────────
    logger.info("\n[6/12] Analyzing vocabulary and TF-IDF …")
    from src.vocabulary import run_vocabulary_analysis
    run_vocabulary_analysis()

    # ─── Step 7: N-gram analysis ──────────────────────────────────────────────
    logger.info("\n[7/12] Extracting n-grams …")
    from src.ngrams import run_ngram_analysis
    run_ngram_analysis()

    if quick:
        logger.info("\n[8-10/12] Skipping embeddings, similarity & topics (--quick mode)")
    else:
        # ─── Step 8: Embeddings ───────────────────────────────────────────────
        logger.info("\n[8/12] Generating embeddings …")
        from src.embeddings import run_embeddings
        run_embeddings()

        # ─── Step 9: Similarity ───────────────────────────────────────────────
        logger.info("\n[9/12] Computing semantic similarity …")
        from src.similarity import run_similarity_analysis
        run_similarity_analysis()

        # ─── Step 10: Topics ──────────────────────────────────────────────────
        logger.info("\n[10/12] Running topic modeling …")
        from src.topics import run_topic_analysis
        run_topic_analysis()

    # ─── Step 11: Statistical tests ───────────────────────────────────────────
    logger.info("\n[11/12] Running statistical tests …")
    from src.statistics import run_statistical_analysis
    run_statistical_analysis()

    # ─── Step 12: Visualization ───────────────────────────────────────────────
    logger.info("\n[12/12] Generating visualizations …")
    from src.visualize import generate_all_figures
    generate_all_figures()

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Results saved to reports/")
    logger.info("  - Figures: reports/figures/")
    logger.info("  - Tables:  reports/tables/")
    logger.info("  - Quality: reports/data_quality_report.md")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review data_quality_report.md")
    logger.info("  2. Inspect sample figures in reports/figures/")
    logger.info("  3. Review statistical_results.csv for significant findings")
    logger.info("  4. Generate final research report (see reports/research_report.md)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Stanford Daily AI vocabulary shift research pipeline"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection (use existing raw data)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip embeddings and topic modeling (saves time)",
    )
    args = parser.parse_args()

    try:
        run_full_pipeline(
            skip_collection=args.skip_collection,
            quick=args.quick,
        )
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Pipeline failed with error: %s", e)
        sys.exit(1)
