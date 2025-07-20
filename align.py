#!/usr/bin/env python3
"""Align bilingual terms using cosine similarity between occurrence embeddings.

This script reads English and Ukrainian occurrence embeddings, groups them by
lemmatized terms, and finds the best matches between language pairs using
maximum cosine similarity between occurrence pairs. Identifies potential
synonyms when multiple Ukrainian groups match the same English group.
Supports parallel processing for faster computation.

Enhanced with:
- Language-specific character filtering (English/Ukrainian)
- Meta score calculation: (avg_en_score + avg_uk_score) / 2 * cosine_similarity
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def has_english_characters(text: str) -> bool:
    """Check if text contains at least one English/Latin character.

    Args:
        text: Text to check.

    Returns:
        True if text contains English characters, False otherwise.
    """
    # Latin alphabet range (Basic Latin + Latin-1 Supplement)
    english_pattern = re.compile(r"[a-zA-Z]")
    return bool(english_pattern.search(text))


def has_ukrainian_characters(text: str) -> bool:
    """Check if text contains at least one Ukrainian/Cyrillic character.

    Args:
        text: Text to check.

    Returns:
        True if text contains Ukrainian characters, False otherwise.
    """
    # Ukrainian Cyrillic characters (including specific Ukrainian letters)
    ukrainian_pattern = re.compile(r"[а-яА-ЯіІїЇєЄҐґ]")
    return bool(ukrainian_pattern.search(text))


class TermOccurrence(NamedTuple):
    """Represents a single term occurrence with its embedding."""

    original_term: str
    occurrence_text: str
    occurrence_index: int
    score: float
    embedding: np.ndarray


class LemmaGroup:
    """Represents a group of occurrences for the same lemmatized term."""

    def __init__(self, lemma: str, language: str) -> None:
        """Initialize lemma group.

        Args:
            lemma: The lemmatized term.
            language: Language code ('en' or 'uk').
        """
        self.lemma = lemma
        self.language = language
        self.occurrences: List[TermOccurrence] = []
        self.avg_score = 0.0
        self.max_score = 0.0
        self.embedding_matrix: Optional[np.ndarray] = None

    def add_occurrence(self, occurrence: TermOccurrence) -> None:
        """Add an occurrence to this group."""
        self.occurrences.append(occurrence)
        self._update_stats()

    def _update_stats(self) -> None:
        """Update group statistics after adding occurrences."""
        if self.occurrences:
            self.avg_score = sum(occ.score for occ in self.occurrences) / len(
                self.occurrences
            )
            self.max_score = max(occ.score for occ in self.occurrences)
            # Stack embeddings for efficient similarity computation
            self.embedding_matrix = np.stack(
                [occ.embedding for occ in self.occurrences]
            )

    def get_occurrence_texts(self) -> List[str]:
        """Get list of occurrence texts in this group."""
        return [occ.occurrence_text for occ in self.occurrences]

    def __len__(self) -> int:
        return len(self.occurrences)

    def __repr__(self) -> str:
        return f"LemmaGroup('{self.lemma}', {self.language}, {len(self.occurrences)} occurrences)"


class AlignmentMatch(NamedTuple):
    """Represents an alignment between English and Ukrainian lemma groups."""

    en_lemma: str
    uk_lemma: str
    similarity_score: float
    en_occurrence: str
    uk_occurrence: str
    en_group_size: int
    uk_group_size: int
    en_avg_score: float
    uk_avg_score: float
    en_max_score: float
    uk_max_score: float
    meta_score: float  # New field: (en_max_score + uk_max_score) / 2 * similarity_score


class BilingualAligner:
    """Aligns bilingual terms using occurrence-level embeddings."""

    def __init__(
        self, similarity_threshold: float = 0.7, enable_filtering: bool = True
    ) -> None:
        """Initialize the bilingual aligner.

        Args:
            similarity_threshold: Minimum similarity score for valid alignments.
            enable_filtering: Whether to enable language-specific character filtering.
        """
        self.similarity_threshold = similarity_threshold
        self.enable_filtering = enable_filtering
        self.en_groups: Dict[str, LemmaGroup] = {}
        self.uk_groups: Dict[str, LemmaGroup] = {}
        self.filtered_stats = {
            "en_total_loaded": 0,
            "en_filtered_out": 0,
            "uk_total_loaded": 0,
            "uk_filtered_out": 0,
        }

    def _should_include_group(self, lemma: str, language: str) -> bool:
        """Check if a lemma group should be included based on character filtering.

        Args:
            lemma: The lemmatized term.
            language: Language code ('en' or 'uk').

        Returns:
            True if the group should be included, False if filtered out.
        """
        if not self.enable_filtering:
            return True

        if language == "en":
            return has_english_characters(lemma)
        elif language == "uk":
            return has_ukrainian_characters(lemma)
        else:
            return True  # Unknown language, include by default

    def load_embeddings_from_jsonl(
        self, jsonl_path: Path, language: str, max_terms: Optional[int] = None
    ) -> None:
        """Load occurrence embeddings and group by lemmatized terms.

        Args:
            jsonl_path: Path to occurrence embeddings JSONL file.
            language: Language code ('en' or 'uk').
            max_terms: Maximum number of unique terms to load.
        """
        groups = self.en_groups if language == "en" else self.uk_groups
        total_loaded = 0
        filtered_out = 0

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)

                    # Skip metadata lines
                    if record.get("type") == "metadata":
                        continue

                    # Extract occurrence data
                    original_term = record["original_term"]
                    occurrence_text = record["occurrence_text"]
                    occurrence_index = record["occurrence_index"]
                    score = record["score"]
                    embedding = np.array(record["embedding"])

                    total_loaded += 1

                    # Apply character filtering
                    if original_term not in groups:
                        if not self._should_include_group(original_term, language):
                            filtered_out += 1
                            continue
                        groups[original_term] = LemmaGroup(original_term, language)

                    # Only add occurrence if the group wasn't filtered out
                    if original_term in groups:
                        # Create occurrence object
                        occurrence = TermOccurrence(
                            original_term=original_term,
                            occurrence_text=occurrence_text,
                            occurrence_index=occurrence_index,
                            score=score,
                            embedding=embedding,
                        )

                        groups[original_term].add_occurrence(occurrence)

                    if max_terms and len(groups) >= max_terms:
                        break

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.error(
                        f"Error processing line {line_num} in {jsonl_path}: {e}"
                    )
                    continue

        # Update filtering statistics
        if language == "en":
            self.filtered_stats["en_total_loaded"] = total_loaded
            self.filtered_stats["en_filtered_out"] = filtered_out
        else:
            self.filtered_stats["uk_total_loaded"] = total_loaded
            self.filtered_stats["uk_filtered_out"] = filtered_out

        logging.info(
            f"Loaded {sum(len(group) for group in groups.values())} occurrences "
            f"for {len(groups)} unique {language.upper()} lemmas from {jsonl_path}"
        )

        if self.enable_filtering and filtered_out > 0:
            logging.info(
                f"Filtered out {filtered_out} {language.upper()} terms without "
                f"proper {language.upper()} characters ({filtered_out/total_loaded*100:.1f}%)"
            )

    def find_best_alignment(
        self, en_group: LemmaGroup, uk_group: LemmaGroup
    ) -> Tuple[float, str, str, float]:
        """Find the best alignment between two lemma groups.

        Args:
            en_group: English lemma group.
            uk_group: Ukrainian lemma group.

        Returns:
            Tuple of (max_similarity, best_en_occurrence, best_uk_occurrence, meta_score).
        """
        # Calculate cosine similarity between all occurrence pairs
        similarities = cosine_similarity(
            en_group.embedding_matrix, uk_group.embedding_matrix
        )

        # Find the best matching pair
        max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
        max_similarity = similarities[max_idx]

        best_en_occurrence = en_group.occurrences[max_idx[0]].occurrence_text
        best_uk_occurrence = uk_group.occurrences[max_idx[1]].occurrence_text

        # Calculate meta score: (avg_en_score + avg_uk_score) / 2 * cosine_similarity
        meta_score = ((en_group.avg_score + uk_group.avg_score) / 2) * float(
            max_similarity
        )

        return float(max_similarity), best_en_occurrence, best_uk_occurrence, meta_score

    def align_terms(
        self,
        min_group_size: int = 1,
        max_alignments_per_en_term: Optional[int] = None,
        num_processes: Optional[int] = None,
    ) -> Tuple[List[AlignmentMatch], Dict[str, List[AlignmentMatch]]]:
        """Align English and Ukrainian terms based on occurrence embeddings.

        Args:
            min_group_size: Minimum number of occurrences required per group.
            max_alignments_per_en_term: Maximum alignments to keep per English term.
            num_processes: Number of processes to use (None for CPU count).

        Returns:
            Tuple of (primary_alignments, potential_synonyms).
        """
        # Set default number of processes
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues

        # Filter groups by minimum size
        valid_en_groups = {
            lemma: group
            for lemma, group in self.en_groups.items()
            if len(group) >= min_group_size
        }
        valid_uk_groups = {
            lemma: group
            for lemma, group in self.uk_groups.items()
            if len(group) >= min_group_size
        }

        logging.info(
            f"Processing {len(valid_en_groups)} English and {len(valid_uk_groups)} "
            f"Ukrainian groups using {num_processes} processes (min_size={min_group_size})"
        )

        if not valid_en_groups or not valid_uk_groups:
            logging.warning("No valid groups found for alignment")
            return [], {}

        # Serialize Ukrainian groups for multiprocessing
        uk_groups_data = {
            lemma: _serialize_lemma_group(group)
            for lemma, group in valid_uk_groups.items()
        }

        # Prepare arguments for workers
        worker_args = []
        for en_lemma, en_group in valid_en_groups.items():
            en_group_data = _serialize_lemma_group(en_group)
            worker_args.append(
                (
                    en_lemma,
                    en_group_data,
                    uk_groups_data,
                    self.similarity_threshold,
                    max_alignments_per_en_term,
                )
            )

        # Process in parallel
        primary_alignments = []
        potential_synonyms = defaultdict(list)

        if num_processes == 1:
            # Single-threaded processing for debugging
            for args in tqdm(worker_args, desc="Aligning terms"):
                en_lemma, group_alignments = _process_english_lemma_worker(args)
                if group_alignments:
                    primary_alignments.append(group_alignments[0])
                    if len(group_alignments) > 1:
                        potential_synonyms[en_lemma] = group_alignments[1:]
        else:
            # Multi-threaded processing
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all tasks
                future_to_lemma = {
                    executor.submit(_process_english_lemma_worker, args): args[0]
                    for args in worker_args
                }

                # Collect results with progress bar
                for future in tqdm(
                    as_completed(future_to_lemma),
                    total=len(future_to_lemma),
                    desc="Aligning terms",
                ):
                    try:
                        en_lemma, group_alignments = future.result()
                        if group_alignments:
                            primary_alignments.append(group_alignments[0])
                            if len(group_alignments) > 1:
                                potential_synonyms[en_lemma] = group_alignments[1:]
                    except Exception as e:
                        lemma = future_to_lemma[future]
                        logging.error(f"Error processing lemma '{lemma}': {e}")

        # Sort primary alignments by meta score (descending)
        primary_alignments.sort(key=lambda x: x.meta_score, reverse=True)

        logging.info(
            f"Found {len(primary_alignments)} primary alignments and "
            f"{len(potential_synonyms)} terms with potential synonyms"
        )

        return primary_alignments, dict(potential_synonyms)

    def save_alignments_jsonl(
        self,
        alignments: List[AlignmentMatch],
        output_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save alignments to JSONL file.

        Args:
            alignments: List of alignment matches to save.
            output_path: Path to output JSONL file.
            metadata: Optional metadata to include.
        """
        with output_path.open("w", encoding="utf-8") as f:
            # Write metadata as first line if provided
            if metadata:
                metadata_record = {"type": "metadata", **metadata}
                f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")

            # Write alignments
            for alignment in alignments:
                record = {
                    "en_lemma": alignment.en_lemma,
                    "uk_lemma": alignment.uk_lemma,
                    "similarity_score": alignment.similarity_score,
                    "en_occurrence": alignment.en_occurrence,
                    "uk_occurrence": alignment.uk_occurrence,
                    "en_group_size": alignment.en_group_size,
                    "uk_group_size": alignment.uk_group_size,
                    "en_avg_score": alignment.en_avg_score,
                    "uk_avg_score": alignment.uk_avg_score,
                    "en_max_score": alignment.en_max_score,
                    "uk_max_score": alignment.uk_max_score,
                    "meta_score": alignment.meta_score,  # Include new meta score
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(f"Saved {len(alignments)} alignments to {output_path}")

    def save_synonyms_jsonl(
        self,
        synonyms: Dict[str, List[AlignmentMatch]],
        output_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save potential synonyms to JSONL file.

        Args:
            synonyms: Dictionary mapping English lemmas to their potential synonym alignments.
            output_path: Path to output JSONL file.
            metadata: Optional metadata to include.
        """
        with output_path.open("w", encoding="utf-8") as f:
            # Write metadata as first line if provided
            if metadata:
                metadata_record = {"type": "metadata", **metadata}
                f.write(json.dumps(metadata_record, ensure_ascii=False) + "\n")

            # Write synonyms
            for en_lemma, synonym_alignments in synonyms.items():
                record = {
                    "en_lemma": en_lemma,
                    "potential_synonyms": [
                        {
                            "uk_lemma": align.uk_lemma,
                            "similarity_score": align.similarity_score,
                            "uk_occurrence": align.uk_occurrence,
                            "uk_group_size": align.uk_group_size,
                            "uk_avg_score": align.uk_avg_score,
                            "meta_score": align.meta_score,  # Include meta score
                        }
                        for align in synonym_alignments
                    ],
                    "synonym_count": len(synonym_alignments),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info(
            f"Saved {len(synonyms)} terms with potential synonyms to {output_path}"
        )

    def _create_histogram(
        self,
        scores: List[float],
        num_bins: int = 40,
        width: int = 60,
        title: str = "Distribution",
    ) -> str:
        """Create a terminal histogram of scores.

        Args:
            scores: List of scores.
            num_bins: Number of histogram bins.
            width: Width of the histogram bars in characters.
            title: Title for the histogram.

        Returns:
            Formatted histogram string.
        """
        if not scores:
            return f"No scores to display histogram for {title}"

        min_score = min(scores)
        max_score = max(scores)

        # Create bins
        bin_width = (max_score - min_score) / num_bins if max_score > min_score else 1
        bins = [0] * num_bins

        # Count scores in each bin
        for score in scores:
            # Handle edge case where score equals max_score
            if max_score > min_score:
                bin_idx = min(int((score - min_score) / bin_width), num_bins - 1)
            else:
                bin_idx = 0
            bins[bin_idx] += 1

        # Find max count for scaling
        max_count = max(bins) if bins else 1

        # Generate histogram
        histogram_lines = []
        histogram_lines.append(f"\n{title} (40 bins):")
        histogram_lines.append("=" * (width + 20))

        for i, count in enumerate(bins):
            # Calculate bin range
            bin_start = min_score + i * bin_width
            bin_end = min_score + (i + 1) * bin_width

            # Create bar
            bar_length = int((count / max_count) * width) if max_count > 0 else 0
            bar = "█" * bar_length

            # Format line
            if count > 0:
                line = f"{bin_start:.3f}-{bin_end:.3f} │{bar:<{width}} │ {count:>4}"
            else:
                line = f"{bin_start:.3f}-{bin_end:.3f} │{'':<{width}} │ {count:>4}"

            histogram_lines.append(line)

        histogram_lines.append("=" * (width + 20))
        histogram_lines.append(f"{'Range':<13} │{'Distribution':<{width}} │ Count")

        return "\n".join(histogram_lines)

    def print_alignment_statistics(
        self,
        alignments: List[AlignmentMatch],
        synonyms: Dict[str, List[AlignmentMatch]],
    ) -> None:
        """Print comprehensive alignment statistics.

        Args:
            alignments: List of primary alignments.
            synonyms: Dictionary of potential synonyms.
        """
        if not alignments:
            print("No alignments found!")
            return

        # Basic statistics
        similarity_scores = [align.similarity_score for align in alignments]
        meta_scores = [align.meta_score for align in alignments]
        en_coverage = len(set(align.en_lemma for align in alignments))
        uk_coverage = len(set(align.uk_lemma for align in alignments))

        print(f"\n{'='*70}")
        print("BILINGUAL ALIGNMENT STATISTICS")
        print(f"{'='*70}")
        print(f"Total English lemmas: {len(self.en_groups)}")
        print(f"Total Ukrainian lemmas: {len(self.uk_groups)}")
        print(f"Primary alignments found: {len(alignments)}")
        print(
            f"English lemmas aligned: {en_coverage} ({en_coverage/len(self.en_groups)*100:.1f}%)"
        )
        print(
            f"Ukrainian lemmas aligned: {uk_coverage} ({uk_coverage/len(self.uk_groups)*100:.1f}%)"
        )
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Processing: {len(alignments)} alignments computed")

        # Filtering statistics
        if self.enable_filtering:
            print(f"\nCharacter Filtering Statistics:")
            en_kept = (
                self.filtered_stats["en_total_loaded"]
                - self.filtered_stats["en_filtered_out"]
            )
            uk_kept = (
                self.filtered_stats["uk_total_loaded"]
                - self.filtered_stats["uk_filtered_out"]
            )
            print(
                f"  English: {en_kept}/{self.filtered_stats['en_total_loaded']} terms kept "
                f"({en_kept/max(1, self.filtered_stats['en_total_loaded'])*100:.1f}%)"
            )
            print(
                f"  Ukrainian: {uk_kept}/{self.filtered_stats['uk_total_loaded']} terms kept "
                f"({uk_kept/max(1, self.filtered_stats['uk_total_loaded'])*100:.1f}%)"
            )

        # Score distributions
        print(f"\nSimilarity Score Distribution:")
        print(f"  Mean: {np.mean(similarity_scores):.3f}")
        print(f"  Median: {np.median(similarity_scores):.3f}")
        print(f"  Min: {np.min(similarity_scores):.3f}")
        print(f"  Max: {np.max(similarity_scores):.3f}")
        print(f"  Std: {np.std(similarity_scores):.3f}")

        print(f"\nMeta Score Distribution:")
        print(f"  Mean: {np.mean(meta_scores):.3f}")
        print(f"  Median: {np.median(meta_scores):.3f}")
        print(f"  Min: {np.min(meta_scores):.3f}")
        print(f"  Max: {np.max(meta_scores):.3f}")
        print(f"  Std: {np.std(meta_scores):.3f}")

        # Terminal histograms
        print(
            self._create_histogram(
                similarity_scores,
                num_bins=40,
                width=50,
                title="Similarity Score Distribution",
            )
        )
        print(
            self._create_histogram(
                meta_scores, num_bins=40, width=50, title="Meta Score Distribution"
            )
        )

        # Synonyms
        if synonyms:
            total_synonym_pairs = sum(len(syns) for syns in synonyms.values())
            print(f"\nPotential Synonyms:")
            print(f"  English terms with synonyms: {len(synonyms)}")
            print(f"  Total synonym pairs: {total_synonym_pairs}")
            print(
                f"  Average synonyms per term: {total_synonym_pairs/len(synonyms):.1f}"
            )

        # Top alignments (sorted by meta score)
        print(f"\nTop 25 Alignments (by Meta Score):")
        print(
            f"{'Rank':<4} {'English':<20} {'Ukrainian':<20} {'Sim':<6} {'Meta':<8} {'Best Pair':<30}"
        )
        print("-" * 90)
        for i, align in enumerate(alignments[:25], 1):
            pair_text = f"{align.en_occurrence} ↔ {align.uk_occurrence}"
            if len(pair_text) > 30:
                pair_text = pair_text[:27] + "..."
            print(
                f"{i:<4} {align.en_lemma:<20} {align.uk_lemma:<20} "
                f"{align.similarity_score:<6.3f} {align.meta_score:<8.1f} {pair_text:<30}"
            )


def _process_english_lemma_worker(
    args_tuple: Tuple,
) -> Tuple[str, List[AlignmentMatch]]:
    """Worker function to process a single English lemma against all Ukrainian lemmas.

    Args:
        args_tuple: Tuple containing (en_lemma, en_group_data, uk_groups_data,
                   similarity_threshold, max_alignments_per_en_term)

    Returns:
        Tuple of (en_lemma, list_of_alignments)
    """
    (
        en_lemma,
        en_group_data,
        uk_groups_data,
        similarity_threshold,
        max_alignments_per_en_term,
    ) = args_tuple

    # Reconstruct English group
    en_group = LemmaGroup(en_lemma, "en")
    for occ_data in en_group_data:
        occurrence = TermOccurrence(
            original_term=occ_data["original_term"],
            occurrence_text=occ_data["occurrence_text"],
            occurrence_index=occ_data["occurrence_index"],
            score=occ_data["score"],
            embedding=np.array(occ_data["embedding"]),
        )
        en_group.add_occurrence(occurrence)

    # Process alignments for this English lemma
    group_alignments = []

    for uk_lemma, uk_group_data in uk_groups_data.items():
        # Reconstruct Ukrainian group
        uk_group = LemmaGroup(uk_lemma, "uk")
        for occ_data in uk_group_data:
            occurrence = TermOccurrence(
                original_term=occ_data["original_term"],
                occurrence_text=occ_data["occurrence_text"],
                occurrence_index=occ_data["occurrence_index"],
                score=occ_data["score"],
                embedding=np.array(occ_data["embedding"]),
            )
            uk_group.add_occurrence(occurrence)

        # Find best alignment between groups
        similarities = cosine_similarity(
            en_group.embedding_matrix, uk_group.embedding_matrix
        )
        max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
        max_similarity = float(similarities[max_idx])

        # Only consider alignments above threshold
        if max_similarity >= similarity_threshold:
            best_en_occurrence = en_group.occurrences[max_idx[0]].occurrence_text
            best_uk_occurrence = uk_group.occurrences[max_idx[1]].occurrence_text

            # Calculate meta score
            meta_score = (
                (en_group.max_score + uk_group.max_score) / 2
            ) * max_similarity

            alignment = AlignmentMatch(
                en_lemma=en_lemma,
                uk_lemma=uk_lemma,
                similarity_score=max_similarity,
                en_occurrence=best_en_occurrence,
                uk_occurrence=best_uk_occurrence,
                en_group_size=len(en_group),
                uk_group_size=len(uk_group),
                en_avg_score=en_group.avg_score,
                uk_avg_score=uk_group.avg_score,
                en_max_score=en_group.max_score,
                uk_max_score=uk_group.max_score,
                meta_score=meta_score,
            )
            group_alignments.append(alignment)

    # Sort by meta score (descending)
    group_alignments.sort(key=lambda x: x.meta_score, reverse=True)

    # Limit alignments if specified
    if max_alignments_per_en_term:
        group_alignments = group_alignments[:max_alignments_per_en_term]

    return en_lemma, group_alignments


def _serialize_lemma_group(group: LemmaGroup) -> List[Dict]:
    """Serialize a lemma group for multiprocessing.

    Args:
        group: LemmaGroup to serialize.

    Returns:
        List of serialized occurrence dictionaries.
    """
    return [
        {
            "original_term": occ.original_term,
            "occurrence_text": occ.occurrence_text,
            "occurrence_index": occ.occurrence_index,
            "score": occ.score,
            "embedding": occ.embedding.tolist(),
        }
        for occ in group.occurrences
    ]


def main() -> None:
    """Main entry point for bilingual term alignment."""
    # Set multiprocessing start method for compatibility
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Start method already set

    parser = argparse.ArgumentParser(
        description="Align bilingual terms using cosine similarity between occurrence embeddings. "
        "Groups occurrences by lemmatized terms and finds best matches between English "
        "and Ukrainian groups using maximum similarity between occurrence pairs. "
        "Includes language-specific character filtering and meta score calculation."
    )
    parser.add_argument(
        "english_embeddings",
        type=Path,
        help="JSONL file with English occurrence embeddings",
    )
    parser.add_argument(
        "ukrainian_embeddings",
        type=Path,
        help="JSONL file with Ukrainian occurrence embeddings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("alignments"),
        help="Output directory for alignment files (default: alignments)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Minimum similarity score for valid alignments (default: 0.90)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=1,
        help="Minimum occurrences per lemma group (default: 1)",
    )
    parser.add_argument(
        "--max-alignments-per-term",
        type=int,
        default=5,
        help="Maximum alignments to keep per English term (default: 5)",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=None,
        help="Max number of terms to draw from each file with embeddings (default: all)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="alignment",
        help="Prefix for output filenames (default: alignment)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores, max 8)",
    )
    parser.add_argument(
        "--disable-filtering",
        action="store_true",
        help="Disable language-specific character filtering",
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

    # Validate input files
    if not args.english_embeddings.exists():
        logging.error(f"English embeddings file not found: {args.english_embeddings}")
        return

    if not args.ukrainian_embeddings.exists():
        logging.error(
            f"Ukrainian embeddings file not found: {args.ukrainian_embeddings}"
        )
        return

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)

    try:
        # Initialize aligner
        enable_filtering = not args.disable_filtering
        aligner = BilingualAligner(
            similarity_threshold=args.similarity_threshold,
            enable_filtering=enable_filtering,
        )

        # Load embeddings
        logging.info("Loading English embeddings...")
        aligner.load_embeddings_from_jsonl(
            args.english_embeddings, "en", max_terms=args.max_terms
        )

        logging.info("Loading Ukrainian embeddings...")
        aligner.load_embeddings_from_jsonl(
            args.ukrainian_embeddings, "uk", max_terms=args.max_terms
        )

        # Perform alignment
        logging.info("Computing bilingual alignments...")
        alignments, synonyms = aligner.align_terms(
            min_group_size=args.min_group_size,
            max_alignments_per_en_term=args.max_alignments_per_term,
            num_processes=args.num_processes,
        )

        # Prepare metadata
        metadata = {
            "english_embeddings_file": str(args.english_embeddings.name),
            "ukrainian_embeddings_file": str(args.ukrainian_embeddings.name),
            "similarity_threshold": args.similarity_threshold,
            "min_group_size": args.min_group_size,
            "max_alignments_per_term": args.max_alignments_per_term,
            "num_processes": args.num_processes or mp.cpu_count(),
            "character_filtering_enabled": enable_filtering,
            "total_alignments": len(alignments),
            "terms_with_synonyms": len(synonyms),
        }

        # Save results
        alignments_file = args.output_dir / f"{args.output_prefix}_primary.jsonl"
        aligner.save_alignments_jsonl(alignments, alignments_file, metadata)

        if synonyms:
            synonyms_file = args.output_dir / f"{args.output_prefix}_synonyms.jsonl"
            aligner.save_synonyms_jsonl(synonyms, synonyms_file, metadata)

        # Print statistics
        aligner.print_alignment_statistics(alignments, synonyms)

        print(f"\nOutput files:")
        print(f"  Primary alignments: {alignments_file}")
        if synonyms:
            print(f"  Potential synonyms: {synonyms_file}")

        num_processes_used = args.num_processes or mp.cpu_count()
        if num_processes_used > 1:
            print(f"\nProcessing completed using {num_processes_used} processes")

        if enable_filtering:
            print(
                f"\nCharacter filtering was enabled - terms without proper language-specific characters were excluded"
            )
        else:
            print(f"\nCharacter filtering was disabled - all terms were processed")

    except KeyboardInterrupt:
        logging.info("Alignment interrupted by user")
    except Exception as e:
        logging.error(f"Alignment failed: {e}")
        raise


if __name__ == "__main__":
    # Multiprocessing protection for Windows and macOS
    mp.freeze_support()
    main()
