# The AI Vocabulary Shift: Linguistic Change in The Stanford Daily, 2015–2026

_Generated 2026-08-12 from pipeline output. All figures below are computed from the collected corpus._

> **Status:** draft skeleton. Numbers are real; interpretation is not written. Sections marked `>>> WRITE:` require the author.


## 1. Corpus

Articles analysed: **N = 11,242**, 2015-01-01 to 2026-08-10, from The Stanford Daily WordPress REST API (`/wp-json/wp/v2/posts`).


### 1.1 Articles by year

```
 year  n_articles  n_opinions  n_news  n_unique_author_ids  mean_word_count  median_word_count  total_words
 2015        1204         448     756                  212            743.4              739.5       895027
 2016        1174         407     767                  203            748.9              711.5       879252
 2017        1072         427     645                  211            809.7              736.0       868016
 2018        1125         368     757                  202            836.5              741.0       941034
 2019        1187         332     855                  229            794.3              705.0       942857
 2020        1446         378    1068                  427            882.5              777.0      1276080
 2021         972         183     789                  285            772.5              713.5       750869
 2022         765         131     634                  184            810.8              725.0       620241
 2023         577         105     472                  148            871.2              785.0       502694
 2024         583          92     491                  150            759.9              711.0       443023
 2025         702         155     547                  137            773.5              720.5       542999
 2026         435         109     326                   86            795.6              756.0       346080
```


**Notes.** 2026 is partial (through 2026-08-10) and is not comparable to full years without adjustment. Distinct-author counts use the numeric `author_id` supplied with each post; author display names could not be retrieved because the publication's `/wp/v2/users/` endpoint requires authentication (HTTP 401), so authors are pseudonymous throughout.


### 1.2 Genre composition of the Opinions corpus

```
 year  n_total  share_column  share_editorial  share_op-ed  share_opinion
 2015      448         0.725            0.027        0.192          0.056
 2016      407         0.590            0.029        0.302          0.079
 2017      427         0.710            0.012        0.173          0.105
 2018      368         0.606            0.038        0.277          0.079
 2019      332         0.346            0.003        0.377          0.274
 2020      378         0.050            0.011        0.455          0.484
 2021      183         0.038            0.000        0.164          0.798
 2022      131         0.023            0.015        0.015          0.947
 2023      105         0.067            0.000        0.019          0.914
 2024       92         0.109            0.011        0.000          0.880
 2025      155         0.245            0.000        0.013          0.742
 2026      109         0.211            0.000        0.000          0.789
```


>>> WRITE: note whether subcategory tagging changed over the window, and whether that reflects editorial practice or metadata practice.


## 2. Methods

- **Text extraction:** HTML → plain text; scripts, navigation, advertising, captions, author bios and related-post blocks removed.

- **Filters:** articles retained at 50–15000 words. Deduplicated by article ID, URL and SHA-256 of cleaned text.

- **Linguistic metrics:** spaCy `en_core_web_sm`. MATTR uses a 100-token moving window and is length-robust; simple TTR and root TTR are length-sensitive and reported for reference only.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.

- **Tests:** Mann–Whitney U with rank-biserial correlation as effect size and percentile bootstrap 95% CIs; fdr_bh correction at α = 0.05.

- **Change-points:** segmented regression, reported against a single-line baseline. A segmented model always fits at least as well as a straight line, so a breakpoint is treated as meaningful only when it clearly beats the linear fit and is well separated from the runner-up year.

- **Control corpus:** the paper's own news reporting, collected and processed identically.


### 2.1 Periodisation

| Period | Years | Role |
|---|---|---|
| Baseline | 2015–2019 | pre-generative-AI |
| Transition | 2020–2023 | excluded from the primary contrast |
| Recent | 2024–2026 | post-widespread-availability |


### 2.2 Authorship classification

Opinion articles were classified as student-authored or guest/institutional using three signals: an explicit title prefix (e.g. 'From the Community'), an institutional byline, and any single byline account holding more than 15% of the Opinions corpus.


>>> WRITE: state the exact counts flagged by each signal and acknowledge that this rule is heuristic.


## 3. Results


### 3.1 Pooled corpus (Opinions + News)

Reported for completeness. These estimates are **confounded** by the changing Opinions/News mix and should not be interpreted on their own.


```
              metric  n_a  n_b  median_a  median_b  effect_size  effect_ci_low  effect_ci_high p_corrected
               mattr 5762 1720    0.7218    0.7310       0.1672         0.1357          0.1990     < 1e-16
                 ttr 5762 1720    0.4698    0.4790       0.0711         0.0425          0.1015     < 0.001
            root_ttr 5762 1720   12.3184   12.5679       0.0773         0.0465          0.1077     < 0.001
         hapax_ratio 5762 1720    0.6953    0.6992       0.0364         0.0074          0.0667       0.033
     func_word_ratio 5762 1720    0.4249    0.4055      -0.3162        -0.3443         -0.2883     < 1e-16
    avg_sentence_len 5762 1720   26.3500   26.1300      -0.0193        -0.0500          0.0091       0.288
    med_sentence_len 5762 1720   25.0000   25.0000      -0.0020        -0.0328          0.0260       0.899
 flesch_reading_ease 5762 1720   43.7075   41.8945      -0.0973        -0.1281         -0.0652     < 0.001
flesch_kincaid_grade 5762 1720   12.4563   12.4579       0.0059        -0.0256          0.0379       0.798
```


### 3.2 Within each corpus (News as control)

```
  corpus               metric  n_pre  n_post  median_pre  median_post  effect_size  effect_ci_low  effect_ci_high p_corrected
opinions                mattr   1982     356      0.7290       0.7490       0.3722         0.3102          0.4291     < 1e-16
opinions                  ttr   1982     356      0.4670       0.4730       0.0814         0.0210          0.1412       0.020
opinions             root_ttr   1982     356     13.1268      14.4421       0.3934         0.3275          0.4520     < 1e-16
opinions          hapax_ratio   1982     356      0.7036       0.7141       0.1352         0.0738          0.1940     < 0.001
opinions      func_word_ratio   1982     356      0.4516       0.4159      -0.4639        -0.5174         -0.4075     < 1e-16
opinions     avg_sentence_len   1982     356     25.5900      24.3350      -0.1599        -0.2214         -0.0961     < 0.001
opinions     med_sentence_len   1982     356     24.0000      23.0000      -0.1422        -0.2052         -0.0823     < 0.001
opinions  flesch_reading_ease   1982     356     45.3018      40.0857      -0.2610        -0.3205         -0.1976     < 0.001
opinions flesch_kincaid_grade   1982     356     12.3060      12.6733       0.0894         0.0267          0.1497       0.011
    news                mattr   3780    1364      0.7183       0.7268       0.1614         0.1260          0.1964     < 1e-16
    news                  ttr   3780    1364      0.4717       0.4805       0.0585         0.0240          0.0931       0.002
    news             root_ttr   3780    1364     11.8686      12.1828       0.0924         0.0576          0.1263     < 0.001
    news          hapax_ratio   3780    1364      0.6904       0.6950       0.0434         0.0087          0.0791       0.022
    news      func_word_ratio   3780    1364      0.4137       0.4024      -0.2025        -0.2373         -0.1709     < 1e-16
    news     avg_sentence_len   3780    1364     26.5800      26.5200      -0.0017        -0.0370          0.0333       0.924
    news     med_sentence_len   3780    1364     25.0000      25.0000       0.0106        -0.0247          0.0437       0.593
    news  flesch_reading_ease   3780    1364     42.9655      42.1984      -0.0349        -0.0718         -0.0002       0.067
    news flesch_kincaid_grade   3780    1364     12.5245      12.4220      -0.0236        -0.0591          0.0126       0.220
```


>>> WRITE: compare Opinions against News for each metric. Note where confidence intervals do not overlap, and where News shows no effect.


### 3.3 Authorship composition

```
 year  n_total  n_guest  n_student  guest_share
 2015      448       81        367        0.181
 2016      407      140        267        0.344
 2017      427      106        321        0.248
 2018      368      108        260        0.294
 2019      332      144        188        0.434
 2020      378       45        333        0.119
 2021      183       80        103        0.437
 2022      131       85         46        0.649
 2023      105       77         28        0.733
 2024       92       60         32        0.652
 2025      155       86         69        0.555
 2026      109       48         61        0.440
```


#### Guest vs student pieces within the same period

Holding the period fixed removes any time trend, so differences here are purely attributable to authorship.


```
        period              metric  n_a  n_b  median_a  median_b  effect_size  effect_ci_low  effect_ci_high p_value
 pre_2015_2019               mattr 1403  579    0.7319    0.7238      -0.1588        -0.2143         -0.1046 < 0.001
 pre_2015_2019     func_word_ratio 1403  579    0.4571    0.4388      -0.2588        -0.3070         -0.2068 < 1e-16
 pre_2015_2019    avg_sentence_len 1403  579   25.5700   25.6500       0.0206        -0.0325          0.0784   0.470
 pre_2015_2019 flesch_reading_ease 1403  579   46.7145   42.4494      -0.1871        -0.2394         -0.1353 < 0.001
 pre_2015_2019            root_ttr 1403  579   13.3758   12.4188      -0.3744        -0.4233         -0.3222 < 1e-16
post_2024_2026               mattr  162  194    0.7566    0.7397      -0.3207        -0.4215         -0.2104 < 0.001
post_2024_2026     func_word_ratio  162  194    0.4153    0.4161      -0.0269        -0.1459          0.0869   0.662
post_2024_2026    avg_sentence_len  162  194   23.9550   24.8300       0.0954        -0.0293          0.2049   0.121
post_2024_2026 flesch_reading_ease  162  194   40.2102   40.0632      -0.0594        -0.1762          0.0685   0.335
post_2024_2026            root_ttr  162  194   14.5660   14.1310      -0.1887        -0.3008         -0.0652   0.002
```


>>> WRITE: state the direction of the authorship difference and whether the change in guest share inflates or suppresses the pre/post contrast.


### 3.4 Student-authored articles only

Primary estimates. Each row applies a progressively stricter control for prolific-author leverage: `2_author_as_unit` gives every author a single value, removing pseudo-replication.


```
           approach              metric  n_pre  n_post  median_pre  median_post  effect_size  effect_ci_low  effect_ci_high p_value
    1_article_level               mattr   1403     162      0.7319       0.7566       0.5181         0.4416          0.5890 < 1e-16
   2_author_as_unit               mattr    218      52      0.7320       0.7589       0.5610         0.4135          0.7026 < 0.001
3_drop_top5_authors               mattr   1141      92      0.7336       0.7557       0.4682         0.3634          0.5744 < 0.001
    1_article_level     func_word_ratio   1403     162      0.4571       0.4153      -0.5495        -0.6221         -0.4714 < 1e-16
   2_author_as_unit     func_word_ratio    218      52      0.4567       0.4242      -0.5183        -0.6544         -0.3700 < 0.001
3_drop_top5_authors     func_word_ratio   1141      92      0.4529       0.4207      -0.4450        -0.5452         -0.3386 < 0.001
    1_article_level    avg_sentence_len   1403     162     25.5700      23.9550      -0.2161        -0.3020         -0.1332 < 0.001
   2_author_as_unit    avg_sentence_len    218      52     25.5556      23.1000      -0.3128        -0.4701         -0.1450 < 0.001
3_drop_top5_authors    avg_sentence_len   1141      92     25.3500      23.5850      -0.2356        -0.3463         -0.1137 < 0.001
    1_article_level flesch_reading_ease   1403     162     46.7145      40.2102      -0.3017        -0.3831         -0.2180 < 0.001
   2_author_as_unit flesch_reading_ease    218      52     46.4346      41.3255      -0.2620        -0.4158         -0.1014   0.003
3_drop_top5_authors flesch_reading_ease   1141      92     46.2008      40.5673      -0.2334        -0.3453         -0.1186 < 0.001
```


#### Author concentration (Opinions)

```
 year  n_articles_with_author  n_unique_authors  articles_per_author  top1_share  top5_share  top10_share   hhi  effective_n_authors
 2015                     448                75                 5.97       0.170       0.380        0.525 0.050                 19.9
 2016                     407                59                 6.90       0.310       0.509        0.651 0.114                  8.8
 2017                     427                58                 7.36       0.192       0.389        0.539 0.058                 17.3
 2018                     368                69                 5.33       0.264       0.402        0.524 0.083                 12.0
 2019                     332                76                 4.37       0.419       0.551        0.657 0.185                  5.4
 2020                     378               219                 1.73       0.085       0.220        0.299 0.016                 61.7
 2021                     183                93                 1.97       0.164       0.317        0.437 0.040                 24.8
 2022                     131                46                 2.85       0.450       0.611        0.710 0.214                  4.7
 2023                     105                28                 3.75       0.486       0.714        0.829 0.255                  3.9
 2024                      92                26                 3.54       0.511       0.685        0.794 0.275                  3.6
 2025                     155                28                 5.54       0.529       0.729        0.858 0.295                  3.4
 2026                     109                30                 3.63       0.358       0.633        0.789 0.158                  6.3
```


>>> WRITE: report how the effective number of authors changed, and treat this as a limitation on how far the later years generalise.


### 3.5 Timing

Year-by-year medians, student-authored articles only:


```
 year   n  mattr  func_word_ratio  avg_sentence_len  flesch_reading_ease
 2015 367 0.7300           0.4466            25.320              44.2851
 2016 267 0.7263           0.4652            26.810              47.0416
 2017 321 0.7349           0.4651            25.390              48.7109
 2018 260 0.7363           0.4586            24.815              49.3153
 2019 188 0.7302           0.4585            26.000              44.2376
 2020 333 0.7281           0.4418            25.340              42.8196
 2021 103 0.7310           0.4476            25.550              44.4157
 2022  46 0.7378           0.4343            27.105              41.4308
 2023  28 0.7252           0.4346            25.550              38.9944
 2024  32 0.7459           0.4148            24.605              39.5524
 2025  69 0.7581           0.4177            24.460              42.0758
 2026  61 0.7632           0.4143            23.340              40.3189
```


#### Unrestricted change-point search

```
             metric  best_breakpoint  best_r2  runner_up  r2_gap  r2_linear linear_p  slope_before  slope_after
              mattr             2023   0.9040       2024  0.0014     0.5079    0.009        0.0006       0.0126
    func_word_ratio             2017   0.9469       2018  0.0147     0.7867  < 0.001        0.0186      -0.0060
   avg_sentence_len             2022   0.7504       2023  0.1244     0.2609    0.090       -0.0586      -0.8620
flesch_reading_ease             2019   0.8716       2018  0.1141     0.6065    0.003        1.6760      -0.5735
```


>>> WRITE: for each metric, state whether the breakpoint precedes or follows late 2022, how well separated it is from the runner-up year, and whether it beats the linear baseline. Metrics that inflect before 2022 cannot be attributed to generative AI.


### 3.6 Article length

Pre/post effects computed within word-count strata (coarsened exact matching), so length is approximately held constant:


```
             metric        length_bin  n_pre  n_post  effect_size  effect_ci_low  effect_ci_high
              mattr        -623 words  397.0    36.0       0.1716        -0.0036          0.3471
              mattr     623-766 words  397.0    37.0       0.3207         0.1190          0.5089
              mattr     766-865 words  396.0    58.0       0.4326         0.2803          0.5779
              mattr    865-1029 words  395.0   101.0       0.3914         0.2794          0.5056
              mattr       1029- words  397.0   124.0       0.4250         0.3035          0.5321
              mattr POOLED (weighted)    NaN     NaN       0.3802            NaN             NaN
           root_ttr        -623 words  397.0    36.0       0.0523        -0.1450          0.2624
           root_ttr     623-766 words  397.0    37.0       0.2270         0.0166          0.4419
           root_ttr     766-865 words  396.0    58.0       0.3673         0.1988          0.5175
           root_ttr    865-1029 words  395.0   101.0       0.3707         0.2532          0.4869
           root_ttr       1029- words  397.0   124.0       0.2995         0.1987          0.4056
           root_ttr POOLED (weighted)    NaN     NaN       0.2982            NaN             NaN
    func_word_ratio        -623 words  397.0    36.0      -0.2918        -0.4687         -0.1180
    func_word_ratio     623-766 words  397.0    37.0      -0.4008        -0.5737         -0.2161
    func_word_ratio     766-865 words  396.0    58.0      -0.3961        -0.5359         -0.2505
    func_word_ratio    865-1029 words  395.0   101.0      -0.5176        -0.6267         -0.4070
    func_word_ratio       1029- words  397.0   124.0      -0.5974        -0.6846         -0.5129
    func_word_ratio POOLED (weighted)    NaN     NaN      -0.4906            NaN             NaN
   avg_sentence_len        -623 words  397.0    36.0      -0.0312        -0.2354          0.1525
   avg_sentence_len     623-766 words  397.0    37.0      -0.1428        -0.3308          0.0461
   avg_sentence_len     766-865 words  396.0    58.0      -0.1941        -0.3475         -0.0279
   avg_sentence_len    865-1029 words  395.0   101.0      -0.3629        -0.4745         -0.2477
   avg_sentence_len       1029- words  397.0   124.0      -0.1346        -0.2545         -0.0180
   avg_sentence_len POOLED (weighted)    NaN     NaN      -0.1995            NaN             NaN
flesch_reading_ease        -623 words  397.0    36.0      -0.3020        -0.4709         -0.1329
flesch_reading_ease     623-766 words  397.0    37.0      -0.2235        -0.4350         -0.0123
flesch_reading_ease     766-865 words  396.0    58.0      -0.1976        -0.3470         -0.0529
flesch_reading_ease    865-1029 words  395.0   101.0      -0.1880        -0.3086         -0.0598
flesch_reading_ease       1029- words  397.0   124.0      -0.3251        -0.4394         -0.2098
flesch_reading_ease POOLED (weighted)    NaN     NaN      -0.2525            NaN             NaN
```


### 3.7 Exploratory AI-associated vocabulary

A list of 59 terms drawn from public commentary about LLM writing style. **This is not a validated AI detector**, and results are hypothesis-generating only.


```
 year   news  opinions  opinions_minus_news
 2015  676.1     909.1                233.0
 2016  664.8     822.5                157.8
 2017  655.9     769.6                113.7
 2018  714.8     825.6                110.8
 2019  787.6    1082.8                295.2
 2020  636.2     853.9                217.7
 2021  793.4     870.9                 77.4
 2022  792.2     983.6                191.5
 2023  758.8     658.4               -100.4
 2024 1000.0    1105.3                105.2
 2025 1140.9    1074.8                -66.1
 2026  796.0    1525.0                729.0
```


>>> WRITE: state whether any rise is specific to opinion writing. If News rises equally, the pattern is not opinion-specific. Note the gap between corpora over time.


## 4. Interpretation

>>> WRITE: This section is deliberately empty.

Constraints to respect:

1. This is an **observational** study. Report temporal association; do not claim causation.
2. Do not treat 2022 as evidence of AI influence by itself. Any coinciding change in corpus size, authorship or editorial policy is an equally consistent explanation.
3. Statistical significance is not importance. Report n, effect size and CI together; large samples make trivial differences significant.
4. Where metrics disagree on timing, say so rather than reporting only the ones that align.
5. The vocabulary list is exploratory and cannot establish that any text was AI-generated.


## 5. Limitations

- **No causal identification.** Temporal coincidence is not causation, and no AI-usage measure exists for these authors.

- **Confounded timing.** Any contraction of the opinion section coincides with generative-AI availability; selection into who kept writing cannot be separated from an AI effect in this design.

- **Author names unavailable.** The WordPress users endpoint returns HTTP 401, so analysis uses numeric `author_id` only. Individual authors cannot be verified as students.

- **Heuristic authorship classification.** Guest content is identified by title prefix, byline and volume, not by verified affiliation.

- **Small recent samples.** Student-authored opinion counts in later years are low; yearly medians there are unstable.

- **2026 is partial** (through 2026-08-10).

- **Embedding truncation.** `all-MiniLM-L6-v2` encodes at most 256 word-pieces, so similarity results describe article openings rather than whole articles.

- **Length-sensitive metrics.** Root TTR scales with article length and is not used for inference.

- **Archive coverage.** Only articles exposed by the site's API are included; category tagging practices changed over the window.


## 6. Reproducibility

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pytest tests/ -v
python -m src.run_pipeline
python -m src.diagnose_confounds
python -m src.diagnose_robustness
python -m src.diagnose_authorship
python -m src.diagnose_final
python -m src.report
```


All parameters are centralised in `src/config.py`. Random seeds are fixed. Figures are in `reports/figures/`; tables in `reports/tables/`.
