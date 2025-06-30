
```bash
# Validate the list of domains, identified by their ids and retrieve the list of entities under this domain which are available in both, english and ukrainian
python validate_domains.py domains.csv

# Download texts from ukrainian and english wikipedia, found for the entities, collected by validate_domains.py. Use your personal token from wi
python corpora_downloader.py domain_entities/aircraft.csv --bearer-token ...

# Combine all downloaded texts for domain/language and run cvalue term extraction for the language.
python ate_it.py extracted_texts/aircraft_uk.jsonl --allow-single-word --language uk --max-text-length 3000_000 --n-max 4 --save-occurrences

# Create term embeddings for term occurences using LaBSE and save them
python create_term_embeddings.py extracted_terms/aircraft_uk_cvalue_terms.occurrences.jsonl
python create_term_embeddings.py extracted_terms/aircraft_en_cvalue_terms.occurrences.jsonl
```