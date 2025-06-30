#!/usr/bin/env python3
"""Create embeddings for extracted terms using sentence transformers.

This script reads term files (TSV format) or term occurrence files (JSONL format)
containing extracted terms and their scores/occurrences, generates embeddings using
multilingual models like LaBSE, and saves the results as JSONL files for efficient
storage and retrieval. Supports creating embeddings for individual term occurrences.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TermEmbedder:
    """Creates embeddings for terms using sentence transformer models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/LaBSE",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the term embedder with a sentence transformer model.

        Args:
            model_name: Name or path of the sentence transformer model.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
            batch_size: Batch size for processing terms.
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logging.info(f"Using device: {device}")

        # Load the model
        logging.info(f"Loading model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            logging.info(
                f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise

    def load_terms_from_tsv(self, tsv_path: Path) -> List[Tuple[str, float]]:
        """Load terms and scores from TSV file.

        Args:
            tsv_path: Path to TSV file with columns 'term' and 'score'.

        Returns:
            List of (term, score) tuples.
        """
        terms = []

        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row_num, row in enumerate(reader, 1):
                try:
                    term = row["term"].strip()
                    score = float(row["score"])

                    if term:  # Skip empty terms
                        terms.append((term, score))
                    else:
                        logging.warning(f"Empty term on row {row_num}")

                except (KeyError, ValueError) as e:
                    logging.error(f"Error processing row {row_num}: {e}")

        logging.info(f"Loaded {len(terms)} terms from {tsv_path}")
        return terms

    def load_terms_from_occurrences_jsonl(
        self, jsonl_path: Path
    ) -> List[Tuple[str, str, float, int]]:
        """Load terms and their occurrences from JSONL file.

        Expected input format:
        {"term": "літак", "score": 1969.0, "occurrence_count": 10,
         "occurrences": ["літаку", "літакові", "літаків", ...]}

        Args:
            jsonl_path: Path to JSONL file with term occurrences.

        Returns:
            List of (original_term, occurrence_text, score, occurrence_index) tuples.
        """
        term_occurrences = []
        total_unique_terms = 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    record = json.loads(line)

                    # Skip metadata lines
                    if record.get("type") == "metadata":
                        continue

                    original_term = record["term"]
                    score = float(record["score"])
                    occurrences = record.get("occurrences", [])
                    occurrence_count = record.get("occurrence_count", len(occurrences))

                    logging.debug(
                        f"Processing term '{original_term}' with {len(occurrences)} occurrences"
                    )

                    # Create entry for each occurrence
                    for idx, occurrence_text in enumerate(occurrences):
                        if (
                            occurrence_text and occurrence_text.strip()
                        ):  # Skip empty occurrences
                            term_occurrences.append(
                                (original_term, occurrence_text.strip(), score, idx)
                            )

                    total_unique_terms += 1

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.error(
                        f"Error processing line {line_num} in {jsonl_path}: {e}"
                    )
                    continue

        logging.info(
            f"Loaded {len(term_occurrences)} term occurrences from {total_unique_terms} "
            f"unique terms in {jsonl_path}"
        )
        return term_occurrences

    def create_embeddings(
        self, terms: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """Create embeddings for a list of terms using batch processing.

        Args:
            terms: List of terms to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            NumPy array of embeddings with shape (n_terms, embedding_dim).
        """
        logging.info(
            f"Creating embeddings for {len(terms)} terms in batches of {self.batch_size}"
        )

        all_embeddings = []

        # Process terms in batches
        with tqdm(
            total=len(terms), disable=not show_progress, desc="Creating embeddings"
        ) as pbar:
            for i in range(0, len(terms), self.batch_size):
                batch_terms = terms[i : i + self.batch_size]

                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.model.encode(
                        batch_terms,
                        batch_size=len(batch_terms),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,  # Normalize for cosine similarity
                    )

                    all_embeddings.append(batch_embeddings)
                    pbar.update(len(batch_terms))

                except Exception as e:
                    logging.error(
                        f"Error processing batch {i//self.batch_size + 1}: {e}"
                    )
                    # Create zero embeddings for failed batch
                    embedding_dim = self.model.get_sentence_embedding_dimension()
                    zero_embeddings = np.zeros((len(batch_terms), embedding_dim))
                    all_embeddings.append(zero_embeddings)
                    pbar.update(len(batch_terms))

        # Concatenate all batch embeddings
        embeddings = np.vstack(all_embeddings)
        logging.info(f"Created embeddings with shape: {embeddings.shape}")

        return embeddings

    def save_embeddings_jsonl(
        self,
        terms_scores: List[Tuple[str, float]],
        embeddings: np.ndarray,
        output_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save terms, scores, and embeddings to JSONL file.

        Args:
            terms_scores: List of (term, score) tuples.
            embeddings: NumPy array of embeddings.
            output_path: Path to output JSONL file.
            metadata: Optional metadata to include in the file.
        """
        if len(terms_scores) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(terms_scores)} terms but {len(embeddings)} embeddings"
            )

        with output_path.open("w", encoding="utf-8") as f:
            # Write metadata as first line if provided
            if metadata:
                metadata_record = {"type": "metadata", **metadata}
                f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")

            # Write term embeddings
            for i, (term, score) in enumerate(terms_scores):
                record = {
                    "term": term,
                    "score": score,
                    "embedding": embeddings[i].tolist(),
                    "embedding_dim": len(embeddings[i]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(f"Saved {len(terms_scores)} term embeddings to {output_path}")

    def save_occurrence_embeddings_jsonl(
        self,
        term_occurrences: List[Tuple[str, str, float, int]],
        embeddings: np.ndarray,
        output_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save term occurrences, scores, and embeddings to JSONL file.

        Args:
            term_occurrences: List of (original_term, occurrence_text, score, occurrence_index) tuples.
            embeddings: NumPy array of embeddings.
            output_path: Path to output JSONL file.
            metadata: Optional metadata to include in the file.
        """
        if len(term_occurrences) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(term_occurrences)} occurrences but {len(embeddings)} embeddings"
            )

        with output_path.open("w", encoding="utf-8") as f:
            # Write metadata as first line if provided
            if metadata:
                metadata_record = {"type": "metadata", **metadata}
                f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")

            # Write occurrence embeddings
            for i, (original_term, occurrence_text, score, occurrence_idx) in enumerate(
                term_occurrences
            ):
                record = {
                    "original_term": original_term,
                    "occurrence_text": occurrence_text,
                    "occurrence_index": occurrence_idx,
                    "score": score,
                    "embedding": embeddings[i].tolist(),
                    "embedding_dim": len(embeddings[i]),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(
            f"Saved {len(term_occurrences)} occurrence embeddings to {output_path}"
        )

    def process_term_file(
        self, input_path: Path, output_path: Path, max_terms: Optional[int] = None
    ) -> None:
        """Process a complete term file from TSV to embeddings JSONL.

        Args:
            input_path: Path to input TSV file.
            output_path: Path to output JSONL file.
            max_terms: Maximum number of terms to process (None for all).
        """
        # Load terms from TSV
        terms_scores = self.load_terms_from_tsv(input_path)

        if not terms_scores:
            logging.error("No terms found in input file")
            return

        # Limit number of terms if specified
        if max_terms and len(terms_scores) > max_terms:
            logging.info(f"Limiting to top {max_terms} terms")
            terms_scores = terms_scores[:max_terms]

        # Extract just the terms for embedding
        terms = [term for term, _ in terms_scores]

        # Create embeddings
        start_time = time.time()
        embeddings = self.create_embeddings(terms)
        embedding_time = time.time() - start_time

        # Prepare metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "total_terms": len(terms_scores),
            "embedding_time_seconds": embedding_time,
            "device": self.device,
            "normalized": True,
            "source_file": str(input_path.name),
            "data_type": "terms",
        }

        # Save results
        self.save_embeddings_jsonl(terms_scores, embeddings, output_path, metadata)

    def process_occurrences_file(
        self, input_path: Path, output_path: Path, max_occurrences: Optional[int] = None
    ) -> None:
        """Process occurrences file from JSONL to embeddings JSONL.

        Args:
            input_path: Path to input occurrences JSONL file.
            output_path: Path to output embeddings JSONL file.
            max_occurrences: Maximum number of occurrences to process (None for all).
        """
        # Load term occurrences from JSONL
        term_occurrences = self.load_terms_from_occurrences_jsonl(input_path)

        if not term_occurrences:
            logging.error("No term occurrences found in input file")
            return

        # Limit number of occurrences if specified
        original_count = len(term_occurrences)
        if max_occurrences and len(term_occurrences) > max_occurrences:
            logging.info(
                f"Limiting to first {max_occurrences} occurrences (from {original_count})"
            )
            term_occurrences = term_occurrences[:max_occurrences]

        # Extract occurrence texts for embedding
        occurrence_texts = [
            occurrence_text for _, occurrence_text, _, _ in term_occurrences
        ]

        # Validate that we have unique occurrence texts
        unique_occurrences = set(occurrence_texts)
        logging.info(
            f"Processing {len(occurrence_texts)} occurrences "
            f"({len(unique_occurrences)} unique forms)"
        )

        # Create embeddings
        start_time = time.time()
        embeddings = self.create_embeddings(occurrence_texts)
        embedding_time = time.time() - start_time

        # Count unique terms
        unique_terms = set(original_term for original_term, _, _, _ in term_occurrences)

        # Prepare metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "total_occurrences": len(term_occurrences),
            "unique_occurrence_forms": len(unique_occurrences),
            "unique_lemmatized_terms": len(unique_terms),
            "embedding_time_seconds": embedding_time,
            "device": self.device,
            "normalized": True,
            "source_file": str(input_path.name),
            "data_type": "occurrences",
        }

        # Save results
        self.save_occurrence_embeddings_jsonl(
            term_occurrences, embeddings, output_path, metadata
        )


def main() -> None:
    """Main entry point for term embedding generation."""
    parser = argparse.ArgumentParser(
        description="Create embeddings for extracted terms using sentence transformers. "
        "Processes TSV files containing terms and scores, or JSONL files containing "
        "term occurrences, and generates embeddings using multilingual models."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="TSV or JSONL file containing terms and scores/occurrences",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output JSONL file path (default: input_file with _embeddings.jsonl extension)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/LaBSE",
        help="Sentence transformer model name (default: LaBSE)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for embeddings (default: auto)",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        help="Maximum number of terms to process from TSV files (default: all)",
    )
    parser.add_argument(
        "--max-occurrences",
        type=int,
        help="Maximum number of occurrences to process from JSONL files (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("term_embeddings"),
        help="Output directory for embeddings (default: term_embeddings)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
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

    # Detect input file type based on file extension and content
    def detect_input_type(file_path: Path) -> str:
        """Detect if input file contains terms (TSV) or occurrences (JSONL)."""
        if file_path.suffix.lower() == ".tsv":
            return "terms"
        elif file_path.suffix.lower() == ".jsonl":
            if "occurrences" in file_path.name.lower():
                return "occurrences"

            # Check file content to determine type
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        record = json.loads(first_line)
                        if "occurrences" in record and "occurrence_count" in record:
                            return "occurrences"
                        elif "term" in record and "score" in record:
                            return "terms"
            except (json.JSONDecodeError, KeyError):
                pass

            return "jsonl"  # Generic JSONL
        else:
            return "unknown"

    input_type = detect_input_type(args.input_file)
    logging.info(f"Detected input type: {input_type}")

    if input_type == "unknown":
        logging.error(f"Unable to determine input file type for {args.input_file}")
        logging.error("Supported formats: .tsv (terms), .jsonl (occurrences or terms)")
        return

    # Generate output filename if not specified
    if args.output_file is None:
        base_name = args.input_file.stem

        if input_type == "occurrences":
            # Remove .occurrences suffix if present
            if base_name.endswith(".occurrences"):
                base_name = base_name[:-12]  # Remove '.occurrences'
            output_filename = f"{base_name}_occurrence_embeddings.jsonl"
        elif input_type == "terms":
            output_filename = f"{base_name}_term_embeddings.jsonl"
        else:  # Generic JSONL
            output_filename = f"{base_name}_embeddings.jsonl"

        args.output_file = args.output_dir / output_filename
    else:
        # Ensure output file is in the output directory
        args.output_file = args.output_dir / args.output_file.name

    # Handle device selection
    device = None if args.device == "auto" else args.device

    try:
        # Initialize embedder
        embedder = TermEmbedder(
            model_name=args.model_name, device=device, batch_size=args.batch_size
        )

        # Process the appropriate file type
        if input_type == "occurrences":
            logging.info("Processing JSONL file with term occurrences")
            embedder.process_occurrences_file(
                args.input_file, args.output_file, max_occurrences=args.max_occurrences
            )
            processing_type = "occurrences"
            max_processed = args.max_occurrences
        elif input_type == "terms":
            if args.input_file.suffix.lower() == ".tsv":
                logging.info("Processing TSV file with terms and scores")
            else:
                logging.info("Processing JSONL file with terms (treating as TSV-like)")
            embedder.process_term_file(
                args.input_file, args.output_file, max_terms=args.max_terms
            )
            processing_type = "terms"
            max_processed = args.max_terms
        else:
            logging.error(f"Unsupported input type: {input_type}")
            return

        # Print summary
        print(f"\n{'='*60}")
        print("TERM EMBEDDING RESULTS")
        print(f"{'='*60}")
        print(f"Input file: {args.input_file}")
        print(f"Input type: {input_type}")
        print(f"Output file: {args.output_file}")
        print(f"Model: {args.model_name}")
        print(f"Device: {embedder.device}")
        print(f"Batch size: {args.batch_size}")
        print(
            f"Embedding dimension: {embedder.model.get_sentence_embedding_dimension()}"
        )

        if input_type == "occurrences":
            # For occurrences, show more detailed statistics
            with args.output_file.open("r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.strip():
                    try:
                        metadata = json.loads(first_line)
                        if metadata.get("type") == "metadata":
                            print(
                                f"Unique lemmatized terms: {metadata.get('unique_lemmatized_terms', 'N/A')}"
                            )
                            print(
                                f"Unique occurrence forms: {metadata.get('unique_occurrence_forms', 'N/A')}"
                            )
                            print(
                                f"Total occurrences processed: {metadata.get('total_occurrences', 'N/A')}"
                            )
                        else:
                            print(
                                f"Occurrences processed: {len(occurrence_texts) if 'occurrence_texts' in locals() else 'N/A'}"
                            )
                    except json.JSONDecodeError:
                        print(
                            f"Occurrences processed: {max_processed if max_processed else 'All'}"
                        )
        else:
            print(
                f"Terms processed: {max_processed if max_processed else 'All from input'}"
            )

        print(f"Embeddings saved to: {args.output_file}")

    except KeyboardInterrupt:
        logging.info("Embedding generation interrupted by user")
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
