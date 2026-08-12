# The AI Vocabulary Shift: Stanford Daily 12-Year Analysis

**Author:** Saidzhan Saitov  
**Research Project:** Independent Research, 2026  
**Repository:** [GitHub](https://github.com/SigmaSaid/Stanford-Daily-Research-Saidzhan)

A comprehensive research project investigating linguistic change in The Stanford Daily from 2015–2026, with focus on the emergence of generative AI.

## Research Question

Did the linguistic evolution of student writing in The Stanford Daily change after generative AI became widely available?

## Key Secondary Questions

1. Did vocabulary diversity change from 2015 to 2026?
2. Did the frequency of particular words and phrases change?
3. Did common n-grams change?
4. Did sentence structure change?
5. Did readability change?
6. Did semantic similarity between articles increase?
7. Did articles become more linguistically homogeneous?
8. Did any major linguistic change occur around 2022–2023?
9. Are apparent AI-associated vocabulary changes actually specific to student opinion writing?

## Important Methodological Notes

- **Correlation vs. Causation**: This is an observational study. We do NOT claim AI caused changes; only that changes coincide with AI's emergence.
- **2026 is incomplete**: Labeled "YTD" in all analyses; never compared to full years without adjustment.
- **Periodization**:
  - 2015–2019: Pre-AI baseline
  - 2020–2022: LLM transition period
  - 2023: Transition year
  - 2024–2026: Generative AI era
- **Topic control**: We investigate whether observed vocabulary shifts are driven by topic changes.
- **Confounders**: COVID-19, changing editorial policies, student demographics, publication volume all investigated.

## Project Structure

```
project/
├── data/
│   ├── raw/                    # Raw JSONL before any processing
│   ├── processed/              # Clean CSV datasets
│   └── embeddings/             # Cached embeddings and metadata
├── src/
│   ├── config.py               # Centralized configuration
│   ├── collect.py              # Multi-method web scraper
│   ├── metadata.py             # Category/author metadata utilities
│   ├── clean.py                # HTML → text cleaning pipeline
│   ├── quality.py              # Data quality checks
│   ├── linguistic_metrics.py   # Article-level metrics
│   ├── vocabulary.py           # Frequency, TTR, diversity analysis
│   ├── ngrams.py               # N-gram extraction and analysis
│   ├── embeddings.py           # Embedding generation and caching
│   ├── similarity.py           # Semantic similarity analysis
│   ├── topics.py               # Topic modeling and analysis
│   ├── statistics.py           # Statistical tests and change-point detection
│   ├── visualize.py            # Chart generation
│   ├── utils.py                # Shared utilities
├── notebooks/
│   └── exploration.ipynb       # Interactive analysis
├── reports/
│   ├── figures/                # PNG visualizations
│   ├── tables/                 # CSV results tables
│   ├── data_quality_report.md  # Detailed quality assessment
│   └── research_report.md      # Final research write-up
├── tests/
│   ├── test_clean.py
│   ├── test_quality.py
│   └── test_metrics.py
│   └── run_pipeline.py         # Main execution script
├── logs/
│   └── pipeline.log            # Pipeline run log
├── requirements.txt            # Python dependencies
└── setup.sh                    # One-time setup script
```

## Quick Start

### Prerequisites

- Python 3.9+
- ~4 GB disk space for raw data
- ~2 GB RAM for analysis
- Internet connection for scraping

### Installation

```bash
# Clone/download the project and enter directory
cd stanford-daily-research

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies (includes tabulate, required by quality.py)
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download sentence transformer model (will auto-download on first use)
```

### Data Collection

```bash
# Run the complete pipeline (always from the project root)
python -m src.run_pipeline

# Or run individual steps (always from the project root):
python -m src.collect              # Scrape Stanford Daily
python -m src.clean                # Clean HTML → text
python -m src.quality              # Generate quality report
python -m src.linguistic_metrics   # Calculate linguistic features
python -m src.vocabulary           # Analyze vocabulary shifts
python -m src.ngrams               # Extract n-grams
python -m src.embeddings           # Generate embeddings
python -m src.similarity           # Semantic similarity analysis
python -m src.topics               # Topic modeling
python -m src.statistics           # Statistical tests
python -m src.visualize            # Generate all figures
```

### Configuration

Edit `src/config.py` to customize:
- Date range (default: 2015-01-01 to 2026-08-10)
- Category IDs (auto-discovered)
- Rate limits and timeouts
- Article length filters
- Embedding model
- Number of embedding samples

## Data Collection Strategy

### Priority Hierarchy

1. **WordPress REST API** — primary method
   - Endpoint: `https://stanforddaily.com/wp-json/wp/v2/posts`
   - Pagination via `per_page` and `page` parameters
   - Category filtering via `categories` parameter
   - Date filtering via `after` and `before` parameters

2. **Archive/Category Pages** — fallback
   - URLs: `/category/opinions/`, `/category/news/`
   - HTML parsing to extract article links
   - Individual article fetching

3. **Individual Article Pages** — final fallback
   - Direct HTML parsing for content

### Features

- **Resumable**: Checks for existing data; continues where it left off
- **Rate-limited**: Configurable delays between requests
- **Cached**: Previously fetched articles not re-downloaded
- **Logged**: All collection activity recorded
- **Error-handling**: Retries transient failures; logs permanent failures
- **Timeout protection**: Configurable request timeouts

## Analysis Pipeline

### 1. Data Quality (quality.py)
- Article count by year/category
- Missing value detection
- Duplicate identification (ID, URL, content)
- Article length distribution
- Author concentration

### 2. Linguistic Metrics (linguistic_metrics.py)
- Word count, sentence count
- Average/median sentence length
- Character count
- Unique word count
- Type-token ratio (MATTR)
- Readability scores (Flesch Reading Ease, Flesch-Kincaid Grade)

### 3. Vocabulary Analysis (vocabulary.py)
- Word frequency by year (per 1M words)
- TF-IDF analysis
- Top changing terms
- Exploratory "AI-associated" vocabulary tracking

### 4. N-gram Analysis (ngrams.py)
- Bigram/trigram extraction
- Frequency normalization
- Top n-grams per year
- Phrase frequency change

### 5. Embedding Analysis (embeddings.py)
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Caching to avoid regeneration

### 6. Semantic Similarity (similarity.py)
- Within-year similarity distributions
- Investigates homogeneity hypothesis

### 7. Topic Modeling (topics.py)
- NMF topic modeling
- Topic distributions over time
- Investigates topic-shift confounding

### 8. Statistical Analysis (statistics.py)
- Mann-Whitney U tests
- Change-point detection (segmented regression, reported against a single-line baseline)
- Multiple comparison corrections (via statsmodels)
- Effect size calculations

### 9. Visualization (visualize.py)
- 15 publication-quality figure files (see list below)
- Clear labels, legends, units

## Output Data Products

### Raw Data
- `data/raw/articles_raw.jsonl` — One article per line, preserved HTML

### Processed Data
- `data/processed/articles.csv` — Cleaned dataset with metadata
- `data/processed/articles_deduplicated.csv` — After deduplication
- `data/embeddings/embeddings.npy` — Dense embedding vectors
- `data/embeddings/article_ids.csv` — Embedding metadata

### Reports
- `reports/data_quality_report.md` — Detailed quality assessment
- `reports/tables/statistical_results.csv` — Test results with n, effect sizes and bootstrap CIs
- `reports/research_report.md` — Final write-up

### Visualizations (`reports/figures/`)

The pipeline generates 15 figure files (13 plot routines; `06_` and `10_` each emit two):

1. `01_articles_by_year` — Articles collected by year
2. `02_word_count_trends` — Average article length by year (mean & median)
3. `03_mattr_by_year` — Vocabulary diversity (MATTR) over time
4. `04_mattr_distribution` — MATTR distribution by period (violin plot)
5. `05_sentence_length` — Median sentence length over time
6. `06_flesch_reading_ease` / `06_flesch_kincaid_grade` — Readability trends
7. `07_ai_vocab_trends` — Aggregate AI-associated term frequency over time
8. `08_ai_vocab_individual` — Top individual AI-associated words over time
9. `09_vocab_changes` — Top words increasing/decreasing in frequency
10. `10_ngram_changes_n2` / `10_ngram_changes_n3` — Top bigram/trigram changes
11. `11_semantic_similarity` — Within-year semantic similarity
12. `12_embedding_pca` — PCA projection of article embeddings
13. `13_topic_distributions` — Topic distribution over time (NMF, stacked area)

### Tables (`reports/tables/`)
- `yearly_corpus_statistics.csv` — Article count, unique authors, etc.
- `linguistic_metrics.csv` — MATTR, readability, sentence length, etc.
- `top_vocabulary_changes.csv` — Words with largest frequency shifts
- `top_ngram_changes.csv` — Phrases with largest frequency shifts
- `ai_associated_vocabulary.csv` — Tracking of exploratory terms
- `semantic_similarity_stats.csv` — Within-year similarity metrics
- `topic_distributions.csv` — Topics over time

## Key Outputs: What You'll Get

After running the pipeline, you'll have:

1. ✅ Complete dataset (raw JSONL + cleaned CSV)
2. ✅ Article-level linguistic metrics
3. ✅ Vocabulary frequency analysis
4. ✅ N-gram trends
5. ✅ Semantic embeddings + within-year similarity analysis
6. ✅ Topic modeling results
7. ✅ Statistical test results with effect sizes
8. ✅ Change-point detection analysis
9. ✅ 15 publication-quality visualization files
10. ✅ Comprehensive research report
11. ✅ Data quality documentation

## Important Limitations & Caveats

- **No causality claims**: Temporal coincidence ≠ causation
- **2026 is incomplete**: Only ~220 days of data
- **Sample size varies**: Early years may have fewer articles
- **Topic confounding**: Vocabulary shifts may reflect topic changes
- **Author effect**: Prolific authors can skew yearly profiles
- **COVID-19 period**: 2020–2021 may show anomalous patterns
- **Archive access**: Only articles accessible via website included
- **Website structure**: Changes in TSD website design over 11 years may affect parsing
- **Embedding truncation**: `all-MiniLM-L6-v2` encodes at most 256 word-pieces (~1,000–1,200
  characters). Most TSD articles are longer, so similarity results describe article
  **openings**, not whole articles. Treat homogeneity findings accordingly.
- **No between-year similarity or dimensionality-reduction (UMAP) figures**: only within-year semantic similarity and a PCA projection are currently implemented in `visualize.py`.

## Testing

```bash
# Run all tests (from the project root)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_clean.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Reproducibility

All results are fully reproducible:
- Random seed set in config
- Package versions pinned as minimum versions in requirements.txt
- Data collection dates logged
- All filtering rules documented
- Code version control ready

## Research Ethics

- No personal data enrichment
- Author names included only as article metadata
- No inference of sensitive characteristics
- Focus on aggregate linguistic patterns
- Public journalism source

## Troubleshooting

### Collection Issues

**Problem**: Scraper stops early
- **Solution**: Check logs in `data/raw/collection.log`. May resume by re-running `collect.py`.

**Problem**: Rate limiting
- **Solution**: Increase delay in `src/config.py` (`REQUEST_DELAY_SECONDS`)

**Problem**: Specific article fails to parse
- **Solution**: Check `data/raw/failed_articles.log`. May need to inspect HTML manually.

### Analysis Issues

**Problem**: Embedding generation is slow
- **Solution**: Reduce sample size in `src/config.py` (`MAX_EMBED_ARTICLES`) or use GPU (if available)

**Problem**: Memory error during similarity calculation
- **Solution**: Use sampling method; see `src/similarity.py`

**Problem**: Topic modeling produces unintelligible topics
- **Solution**: Adjust `NUM_TOPICS` in `src/config.py` or data filters

## Contact & Questions

For questions about methodology, see the research report and docstrings throughout the code.

---

**Last Updated**: 2026-08-11
**Status**: Audited and repaired; synthetic end-to-end validation passing (46 unit tests, full 12-step pipeline exit 0). Live API and embedding-model download NOT yet verified.
**Real Data Status**: Not collected in this environment (network restrictions). Ready to collect on your local machine.
