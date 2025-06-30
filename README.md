# POC of the automated pipeline to create domain-specific bilingual thesauri using wikipedia and wikidata texts

## Installation
```bash
# clone the repo
git clone https://github.com/lang-uk/schmezaurus
cd schmezaurus

# Activate virtual environment and install dependencies
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Download spacy models
spacy download uk_core_news_trf
spacy download en_core_web_trf
```

## Prerequisites
[Create a token](https://api.wikimedia.org/wiki/Authentication#Personal_API_tokens) for wikimedia API to enjoy higher rate limits (100 reqs/h -> 5000 reqs/h).

Pick the domain of your interest from domains.csv (or add your own domain, identified by wikidata id), and remove the rest of domains (keep headers tho!).

## Test run
```bash
# Validate the list of domains, identified by their ids and retrieve the list of entities under this domain which are available in both, english and ukrainian
python validate_domains.py domains.csv
# the results are now stored in domain_entities/

# Download texts from ukrainian and english wikipedia, found for the entities, collected by validate_domains.py. Use your personal token from wikimedia API
python corpora_downloader.py domain_entities/aircraft.csv --bearer-token ...
# the results are saved in extracted_texts/

# Combine all downloaded texts for domain/language and run cvalue term extraction for the language.
# It'll take some time.
python ate_it.py extracted_texts/aircraft_uk.jsonl --allow-single-word --language uk --max-text-length 3000_000 --n-max 4 --save-occurrences
# the results are stored in extracted_terms/

# Create term embeddings for term occurences using LaBSE and store them
python create_term_embeddings.py extracted_terms/aircraft_uk_cvalue_terms.occurrences.jsonl
python create_term_embeddings.py extracted_terms/aircraft_en_cvalue_terms.occurrences.jsonl
# the results are stored in term_embeddings/

# Align the terms through their occurences using cosine similarity. Keep in mind, that execution time is quadratic to the number of terms in --max-terms.
python align.py term_embeddings/aircraft_en_cvalue_terms_occurrence_embeddings.jsonl term_embeddings/aircraft_uk_cvalue_terms_occurrence_embeddings.jsonl --max-terms 2000
# the results are stored in extracted_terms/
```

voila, you should receive something like this:

```
2025-06-30 11:37:27,512 - INFO - Loading English embeddings...
2025-06-30 11:37:28,028 - INFO - Loaded 3162 occurrences for 2000 unique EN lemmas from term_embeddings/aircraft_en_cvalue_terms_occurrence_embeddings.jsonl
2025-06-30 11:37:28,028 - INFO - Loading Ukrainian embeddings...
2025-06-30 11:37:29,197 - INFO - Loaded 5760 occurrences for 2000 unique UK lemmas from term_embeddings/aircraft_uk_cvalue_terms_occurrence_embeddings.jsonl
2025-06-30 11:37:29,197 - INFO - Computing bilingual alignments...
2025-06-30 11:37:29,198 - INFO - Processing 2000 English and 2000 Ukrainian groups using 8 processes (min_size=1)
Aligning terms: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [05:55<00:00,  5.63it/s]
2025-06-30 11:43:25,344 - INFO - Found 431 primary alignments and 44 terms with potential synonyms
2025-06-30 11:43:25,418 - INFO - Saved 431 alignments to alignments/alignment_primary.jsonl
2025-06-30 11:43:25,420 - INFO - Saved 44 terms with potential synonyms to alignments/alignment_synonyms.jsonl

======================================================================
BILINGUAL ALIGNMENT STATISTICS
======================================================================
Total English lemmas: 2000
Total Ukrainian lemmas: 2000
Primary alignments found: 431
English lemmas aligned: 431 (21.6%)
Ukrainian lemmas aligned: 396 (19.8%)
Similarity threshold: 0.95
Processing: 431 alignments computed

Similarity Score Distribution:
  Mean: 0.967
  Median: 0.967
  Min: 0.950
  Max: 1.000
  Std: 0.010

Similarity Score Distribution (40 bins):
======================================================================
0.950-0.951 │██████████████████████████████████████             │   20
0.951-0.953 │██████████████████████████████                     │   16
0.953-0.954 │████████████████████████████████                   │   17
0.954-0.955 │██████████████████████████████████████████████     │   24
0.955-0.956 │████████████████████████████████████               │   19
0.956-0.958 │██████████████████████████                         │   14
0.958-0.959 │████████████████████████████                       │   15
0.959-0.960 │███████████████████████                            │   12
0.960-0.961 │████████████████████████████                       │   15
0.961-0.963 │██████████████████████████                         │   14
0.963-0.964 │████████████████████████████                       │   15
0.964-0.965 │█████████████████                                  │    9
0.965-0.966 │██████████████████████████                         │   14
0.966-0.968 │████████████████████████████                       │   15
0.968-0.969 │██████████████████████████████                     │   16
0.969-0.970 │███████████████████████                            │   12
0.970-0.971 │██████████████████████████████████████████████████ │   26
0.971-0.973 │██████████████████████████████                     │   16
0.973-0.974 │████████████████████████████████████████           │   21
0.974-0.975 │██████████████████████████████████████             │   20
0.975-0.976 │██████████████████████████████████                 │   18
0.976-0.978 │█████████████████████████                          │   13
0.978-0.979 │██████████████████████████████                     │   16
0.979-0.980 │█████████████████                                  │    9
0.980-0.981 │█████████████████████████                          │   13
0.981-0.983 │█████████████████████                              │   11
0.983-0.984 │███████                                            │    4
0.984-0.985 │█████████████                                      │    7
0.985-0.986 │█████████                                          │    5
0.986-0.988 │█                                                  │    1
0.988-0.989 │█                                                  │    1
0.989-0.990 │                                                   │    0
0.990-0.991 │                                                   │    0
0.991-0.993 │                                                   │    0
0.993-0.994 │                                                   │    0
0.994-0.995 │                                                   │    0
0.995-0.996 │                                                   │    0
0.996-0.998 │                                                   │    0
0.998-0.999 │                                                   │    0
0.999-1.000 │█████                                              │    3
======================================================================
Range         │Distribution                                       │ Count

Potential Synonyms:
  English terms with synonyms: 44
  Total synonym pairs: 49
  Average synonyms per term: 1.1

Top 25 Alignments:
Rank English              Ukrainian            Score  Best Pair
--------------------------------------------------------------------------------
1    p.                   p.                   1.000  p. ↔ p.
2    ×                    ×                    1.000  × ↔ ×
3    °                    °                    1.000  ° ↔ °
4    problem              проблема             0.988  problems ↔ проблеми
5    difference           відмінність          0.986  differences ↔ відмінності
6    difficulty           труднощі             0.986  difficulties ↔ труднощі
7    organization         організація          0.986  organization ↔ організація
8    production           виробництво          0.986  production ↔ виробництво
9    result               результат            0.986  results ↔ результати
10   system               система              0.985  system ↔ система
11   technology           технологія           0.985  technologies ↔ технології
12   development          розвиток             0.985  development ↔ розвиток
13   group                група                0.985  group ↔ група
14   photo                фото                 0.985  photo ↔ фото
15   school               школа                0.985  school ↔ школа
16   change               зміна                0.985  changes ↔ зміни
17   museum               музей                0.984  museum ↔ музей
18   hand                 рука                 0.984  hands ↔ руки
19   word                 слово                0.983  words ↔ слова
20   price                ціна                 0.983  prices ↔ ціни
21   president            президент            0.983  president ↔ президент
22   life                 життя                0.983  life ↔ життя
23   mm                   мм                   0.983  mm ↔ мм
24   color                колір                0.982  color ↔ колір
25   second world war     другий світовий війна 0.982  second world war ↔ другої с...

Output files:
  Primary alignments: alignments/alignment_primary.jsonl
  Potential synonyms: alignments/alignment_synonyms.jsonl
```

## Optional fun: visualize embeddings with interactive visualizer and t-SNE (2d or 3d!)

`python visualize_embeddings.py term_embeddings/aircraft_en_cvalue_terms_embeddings.jsonl term_embeddings/aircraft_uk_cvalue_terms_embeddings.jsonl --source-labels "English" "Ukrainian"  --dimensions 2 --max-terms-per-file 1500`
then open the file `embedding_visualization_2files.html`

## Detailed documentation
All the elements of the pipeline and implementation details are disclosed in [Detailed documentation](AUTODOC.md). Use with care, it was auto-generated by LLM and quickly proof-read by me.

