# Data Quality Report

**Generated:** 2026-08-12 13:24


## Summary

| Metric | Value |
| --- | --- |
| total_articles | 11244 |
| missing_title | 0 |
| missing_date | 0 |
| missing_author_name | 11244 |
| missing_text | 0 |
| missing_article_type | 0 |
| missing_year | 0 |
| articles_below_min_words | 0 |
| articles_above_max_words | 0 |
| articles_empty_text | 0 |
| median_word_count | 733.0 |
| mean_word_count | 801.5694592671647 |
| articles_after_dedup | 11242 |
| articles_removed_dedup | 2 |

## Articles by Year

|   year |   n_articles |   n_opinions |   n_news |   n_unique_author_ids |   n_resolved_author_names |   mean_word_count |   median_word_count |      total_words |
|-------:|-------------:|-------------:|---------:|----------------------:|--------------------------:|------------------:|--------------------:|-----------------:|
|   2015 |         1204 |          448 |      756 |                   212 |                         0 |             743.4 |               739.5 | 895027           |
|   2016 |         1174 |          407 |      767 |                   203 |                         0 |             748.9 |               711.5 | 879252           |
|   2017 |         1072 |          427 |      645 |                   211 |                         0 |             809.7 |               736   | 868016           |
|   2018 |         1125 |          368 |      757 |                   202 |                         0 |             836.5 |               741   | 941034           |
|   2019 |         1187 |          332 |      855 |                   229 |                         0 |             794.3 |               705   | 942857           |
|   2020 |         1446 |          378 |     1068 |                   427 |                         0 |             882.5 |               777   |      1.27608e+06 |
|   2021 |          972 |          183 |      789 |                   285 |                         0 |             772.5 |               713.5 | 750869           |
|   2022 |          765 |          131 |      634 |                   184 |                         0 |             810.8 |               725   | 620241           |
|   2023 |          577 |          105 |      472 |                   148 |                         0 |             871.2 |               785   | 502694           |
|   2024 |          583 |           92 |      491 |                   150 |                         0 |             759.9 |               711   | 443023           |
|   2025 |          702 |          155 |      547 |                   137 |                         0 |             773.5 |               720.5 | 542999           |
|   2026 |          435 |          109 |      326 |                    86 |                         0 |             795.6 |               756   | 346080           |


## Notes

- 2026 is a partial year (YTD). Do not compare directly to full years.
- 'opinions' corpus = Opinions/Op-Ed/Editorial/Column/Letter as tagged by TSD.
- 'news' corpus = News/University/Local/National/World categories.
- Word count filters: min=50, max=15000.