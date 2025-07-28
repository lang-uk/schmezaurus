import csv
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a primariy glossary JSONL file to CSV format."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the primary input JSONL file containing the glossary.",
    )
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")

    args = parser.parse_args()

    glossary = []
    with open(args.input_file, "r", encoding="utf-8") as json_file:
        for line_num, line in enumerate(json_file, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Skip metadata lines
            if record.get("type") == "metadata":
                continue

            glossary.append(record)

    # {"en_lemma": "learning", "uk_lemma": "навчання", "similarity_score": 0.9099790489625272, "en_occurrence": "learnings", "uk_occurrence": "навчанні", "all_en_occurrences": ["learning", "learnings"], "all_uk_occurrences": ["навчання", "навчаннях", "навчанні", "навчанням", "навчанню"], "en_group_size": 2, "uk_group_size": 5, "en_avg_score": 2479.0, "uk_avg_score": 2234.0, "en_max_score": 2479.0, "uk_max_score": 2234.0, "meta_score": 2144.3656288801953}
    with open(args.output_file, "w", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "en_lemma",
                "all_en_occurrences",
                "uk_lemma",
                "all_uk_occurrences",
                "similarity_score",
                "meta_score",
                "en_max_score",
                "uk_max_score",
                "en_lemma_word_count",
                "uk_lemma_word_count",
            ],
        )
        writer.writeheader()

        for record in glossary:
            writer.writerow(
                {
                    "en_lemma": record.get("en_lemma", ""),
                    "all_en_occurrences": ", ".join(
                        record.get("all_en_occurrences", [])
                    ).replace("\n", ""),
                    "uk_lemma": record.get("uk_lemma", ""),
                    "all_uk_occurrences": ", ".join(
                        record.get("all_uk_occurrences", [])
                    ).replace("\n", ""),
                    "similarity_score": record.get("similarity_score", 0.0),
                    "meta_score": record.get("meta_score", 0.0),
                    "en_max_score": record.get("en_max_score", 0.0),
                    "uk_max_score": record.get("uk_max_score", 0.0),
                    "en_lemma_word_count": record.get("en_lemma_word_count", 0),
                    "uk_lemma_word_count": record.get("uk_lemma_word_count", 0),
                }
            )
