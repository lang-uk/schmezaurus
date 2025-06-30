#!/usr/bin/env python3
"""Extract terms from Wikipedia text corpora using various term extraction methods.

This script reads JSONL files containing Wikipedia article texts,
combines them into a single corpus, and applies term extraction
algorithms (Basic, C-Value, ComboBasic) to identify key terms.
Optionally saves detailed occurrence information for each extracted term.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc

# Import term extraction methods from matcha library
from matcha import basic, cvalue, combo_basic


class TermExtractor:
    """Extracts terms from text corpora using various algorithms."""

    def __init__(
        self,
        language: str = "en",
        spacy_model: Optional[str] = None,
        disable_components: Optional[List[str]] = None,
    ) -> None:
        """Initialize the term extractor with a spaCy model.

        Args:
            language: Language code ('en' or 'uk').
            spacy_model: Specific spaCy model name. If None, uses default for language.
            disable_components: List of spaCy pipeline components to disable.
        """
        self.language = language

        # Set default models for each language
        if spacy_model is None:
            model_map = {
                "en": "en_core_web_trf",
                "uk": "uk_core_news_trf",
                "de": "de_dep_news_trf",
            }
            spacy_model = model_map.get(language, "en_core_web_trf")

        # Default components to disable for better performance
        if disable_components is None:
            disable_components = ["parser", "ner"]

        try:
            self.nlp = spacy.load(spacy_model, disable=disable_components)
            self.nlp.max_length = 10_000_000
            logging.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logging.error(
                f"Could not load spaCy model '{spacy_model}'. "
                f"Please install it with: python -m spacy download {spacy_model}"
            )
            raise

    def load_jsonl_texts(self, jsonl_path: Path) -> List[str]:
        """Load and extract text content from JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing Wikipedia articles.

        Returns:
            List of text strings from the articles.
        """
        texts = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())

                    # Extract text content
                    text = record.get("text", "")
                    if text and len(text.strip()) > 100:  # Filter very short texts
                        texts.append(text.strip())
                    else:
                        logging.warning(f"Skipping short text in line {line_num}")

                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error on line {line_num}: {e}")
                except Exception as e:
                    logging.error(f"Error processing line {line_num}: {e}")

        logging.info(f"Loaded {len(texts)} texts from {jsonl_path}")
        return texts

    def combine_texts(self, texts: List[str], max_length: Optional[int] = None) -> str:
        """Combine multiple texts into a single corpus.

        Args:
            texts: List of text strings to combine.
            max_length: Maximum length of combined text (for memory management).

        Returns:
            Combined text string.
        """
        combined = "\n\n".join(texts)

        if max_length and len(combined) > max_length:
            logging.warning(
                f"Truncating combined text from {len(combined)} to {max_length} characters"
            )
            combined = combined[:max_length]

        logging.info(
            f"Combined corpus: {len(combined):,} characters, {len(texts)} documents"
        )
        return combined

    def extract_terms_basic(
        self, doc: Doc, n_min: int = 2, alpha: float = 0.72
    ) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Extract terms using the Basic algorithm.

        Args:
            doc: spaCy Doc object.
            n_min: Minimum term length in tokens.
            alpha: Weight parameter for nesting factor.

        Returns:
            Tuple of (term_scores, term_occurrences).
        """
        return basic(doc, alpha=alpha, n_min=n_min)

    def extract_terms_cvalue(
        self, doc: Doc, n_min: int = 2, smoothing: float = 0.1, n_max: int = 4
    ) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Extract terms using the C-Value algorithm.

        Args:
            doc: spaCy Doc object.
            n_min: Minimum term length in tokens.
            smoothing: Smoothing factor for length calculation.
            n_max: Maximum term length in tokens.

        Returns:
            Tuple of (term_scores, term_occurrences).
        """
        return cvalue(doc, n_min=n_min, smoothing=smoothing, n_max=n_max)

    def extract_terms_combo_basic(
        self, doc: Doc, n_min: int = 2, alpha: float = 0.75, beta: float = 0.1
    ) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Extract terms using the ComboBasic algorithm.

        Args:
            doc: spaCy Doc object.
            n_min: Minimum term length in tokens.
            alpha: Weight for superset terms.
            beta: Weight for subset terms.

        Returns:
            Tuple of (term_scores, term_occurrences).
        """
        return combo_basic(doc, alpha=alpha, beta=beta, n_min=n_min)

    def process_corpus(
        self,
        text: str,
        method: str,
        n_min: int = 2,
        allow_single_word: bool = False,
        **method_kwargs,
    ) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Process text corpus and extract terms using specified method.

        Args:
            text: Input text corpus.
            method: Term extraction method ('basic', 'cvalue', 'combo_basic').
            n_min: Minimum term length in tokens.
            allow_single_word: Whether to allow single-word terms.
            **method_kwargs: Additional arguments for extraction methods.

        Returns:
            Tuple of (term_scores, term_occurrences).
        """
        if allow_single_word:
            n_min = 1

        logging.info(f"Processing corpus with {method} method (n_min={n_min})")

        # Process text with spaCy
        logging.info("Processing text with spaCy...")
        doc = self.nlp(text.lower())
        logging.info(f"Processed {len(doc)} tokens")

        # Apply term extraction method
        if method == "basic":
            return self.extract_terms_basic(doc, n_min=n_min, **method_kwargs)
        elif method == "cvalue":
            smoothing = 1.0 if allow_single_word else 0.1
            return self.extract_terms_cvalue(
                doc, n_min=n_min, smoothing=smoothing, **method_kwargs
            )
        elif method == "combo_basic":
            return self.extract_terms_combo_basic(doc, n_min=n_min, **method_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def save_results(
        self,
        term_scores: Dict[str, float],
        term_occurrences: Dict[str, List],
        output_path: Path,
        top_k: Optional[int] = None,
        save_occurrences: bool = False,
    ) -> None:
        """Save term extraction results to file(s).

        Args:
            term_scores: Dictionary mapping terms to their scores.
            term_occurrences: Dictionary mapping terms to their occurrence spans.
            output_path: Path to output file for scores.
            top_k: Number of top terms to save (None for all).
            save_occurrences: Whether to save occurrence details to separate file.
        """
        # Sort terms by score in descending order
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

        if top_k:
            sorted_terms = sorted_terms[:top_k]

        # Save term scores
        with output_path.open("w", encoding="utf-8") as f:
            f.write("term\tscore\n")
            for term, score in sorted_terms:
                f.write(f"{term}\t{score:.6f}\n")

        logging.info(f"Saved {len(sorted_terms)} terms to {output_path}")

        # Save occurrences if requested
        if save_occurrences:
            occurrences_path = output_path.with_suffix(".occurrences.jsonl")
            self._save_occurrences(term_occurrences, sorted_terms, occurrences_path)

    def _save_occurrences(
        self,
        term_occurrences: Dict[str, List],
        sorted_terms: List[Tuple[str, float]],
        occurrences_path: Path,
    ) -> None:
        """Save term occurrences to JSONL file.

        Args:
            term_occurrences: Dictionary mapping terms to occurrence spans.
            sorted_terms: List of (term, score) tuples in order.
            occurrences_path: Path to occurrences output file.
        """
        with occurrences_path.open("w", encoding="utf-8") as f:
            for term, score in sorted_terms:
                if term in term_occurrences:
                    # Get unique occurrences
                    unique_occurrences = list(set(to.text for to in term_occurrences[term]))

                    # Save record
                    record = {
                        "term": term,
                        "score": score,
                        "occurrence_count": len(unique_occurrences),
                        "occurrences": unique_occurrences,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(f"Saved term occurrences to {occurrences_path}")


def main() -> None:
    """Main entry point for term extraction."""
    parser = argparse.ArgumentParser(
        description="Extract terms from Wikipedia text corpora using various algorithms. "
        "Processes JSONL files containing Wikipedia articles and applies "
        "term extraction methods (Basic, C-Value, ComboBasic). "
        "Optionally saves detailed occurrence information for analysis."
    )
    parser.add_argument(
        "input_file", type=Path, help="JSONL file containing Wikipedia article texts"
    )
    parser.add_argument(
        "--method",
        choices=["basic", "cvalue", "combo_basic"],
        default="cvalue",
        help="Term extraction method (default: cvalue)",
    )
    parser.add_argument(
        "--language",
        choices=["en", "uk", "de"],
        default="en",
        help="Language for spaCy model (default: en)",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        help="Specific spaCy model name (overrides --language)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_terms"),
        help="Directory for output files (default: extracted_terms)",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=2,
        help="Minimum term length in tokens (default: 2)",
    )
    parser.add_argument(
        "--allow-single-word",
        action="store_true",
        help="Allow single-word terms (sets n_min=1)",
    )
    parser.add_argument(
        "--top-k", type=int, help="Number of top terms to save (default: all)"
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=1000000,
        help="Maximum combined text length in characters (default: 1M)",
    )
    parser.add_argument(
        "--save-occurrences",
        action="store_true",
        help="Save term occurrences to separate JSONL file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # Method-specific arguments
    parser.add_argument(
        "--alpha", type=float, help="Alpha parameter for Basic/ComboBasic methods"
    )
    parser.add_argument(
        "--beta", type=float, help="Beta parameter for ComboBasic method"
    )
    parser.add_argument(
        "--smoothing", type=float, help="Smoothing parameter for C-Value method"
    )
    parser.add_argument(
        "--n-max", type=int, help="Maximum term length in tokens for C-Value method"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    try:
        # Initialize term extractor
        extractor = TermExtractor(language=args.language, spacy_model=args.spacy_model)

        # Load texts from JSONL file
        texts = extractor.load_jsonl_texts(args.input_file)

        if not texts:
            logging.error("No texts found in input file")
            return

        # Combine texts into corpus
        combined_text = extractor.combine_texts(texts, max_length=args.max_text_length)

        # Prepare method-specific arguments
        method_kwargs = {}
        if args.alpha is not None:
            method_kwargs["alpha"] = args.alpha
        if args.beta is not None:
            method_kwargs["beta"] = args.beta
        if args.smoothing is not None:
            method_kwargs["smoothing"] = args.smoothing
        if args.n_max is not None:
            method_kwargs["n_max"] = args.n_max

        # Extract terms
        term_scores, term_occurrences = extractor.process_corpus(
            combined_text,
            method=args.method,
            n_min=args.n_min,
            allow_single_word=args.allow_single_word,
            **method_kwargs,
        )

        # Generate output filename
        input_stem = args.input_file.stem
        output_filename = f"{input_stem}_{args.method}_terms.tsv"
        output_path = args.output_dir / output_filename

        # Save results
        extractor.save_results(
            term_scores,
            term_occurrences,
            output_path,
            top_k=args.top_k,
            save_occurrences=args.save_occurrences,
        )

        # Print summary
        print(f"\n{'='*60}")
        print("TERM EXTRACTION RESULTS")
        print(f"{'='*60}")
        print(f"Input file: {args.input_file}")
        print(f"Method: {args.method}")
        print(f"Language: {args.language}")
        print(f"Documents processed: {len(texts)}")
        print(f"Combined text length: {len(combined_text):,} characters")
        print(f"Terms extracted: {len(term_scores)}")
        print(f"Output saved to: {output_path}")

        if args.save_occurrences:
            occurrences_file = output_path.with_suffix(".occurrences.jsonl")
            print(f"Occurrences saved to: {occurrences_file}")

        # Show top 10 terms
        top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 terms:")
        for i, (term, score) in enumerate(top_terms, 1):
            print(f"  {i:2d}. {term:<30} {score:.4f}")

    except KeyboardInterrupt:
        logging.info("Term extraction interrupted by user")
    except Exception as e:
        logging.error(f"Term extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
