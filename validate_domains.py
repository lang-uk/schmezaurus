#!/usr/bin/env python3
"""Validate bilingual thesaurus domains by querying Wikidata for coverage.

This script reads a CSV file containing domain names and Wikidata IDs,
then queries Wikidata to find entities within each domain and their
Ukrainian/English Wikipedia page availability.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import diskcache as dc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DomainStats(NamedTuple):
    """Statistics for a domain's Wikipedia coverage."""

    domain_name: str
    wikidata_id: str
    total_entities: int
    both_languages: int
    coverage_percent: float


class WikidataValidator:
    """Validates domain coverage in Wikidata and Wikipedia."""

    WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

    def __init__(self, cache_dir: Path = Path(".cache"), page_size: int = 1000) -> None:
        """Initialize the validator.

        Args:
            cache_dir: Directory for caching SPARQL query results.
            page_size: Number of results per paginated query.
        """
        self.session = self._create_session()
        self.cache_dir = cache_dir
        self.cache = dc.Cache(str(cache_dir))
        self.output_dir = Path("domain_entities")
        self.output_dir.mkdir(exist_ok=True)
        self.page_size = page_size

    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy for robust API calls."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(
            {"User-Agent": "WikidataValidator/1.0 (Educational Research Tool)"}
        )
        return session

    def _execute_sparql_query(
        self, query: str, cache_key: str, retry_count: int = 3
    ) -> Optional[List[Dict]]:
        """Execute a SPARQL query against Wikidata with caching and error handling.

        Args:
            query: SPARQL query string.
            cache_key: Unique key for caching this query result.
            retry_count: Number of retries for failed requests.

        Returns:
            List of result bindings or None if query failed.
        """
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logging.debug(f"Using cached result for {cache_key}")
            return cached_result

        for attempt in range(retry_count):
            try:
                logging.debug(
                    f"Executing SPARQL query for {cache_key} (attempt {attempt + 1}/{retry_count})"
                )
                response = self.session.get(
                    self.WIKIDATA_ENDPOINT,
                    params={"query": query, "format": "json"},
                    timeout=600,  # Increased timeout for large queries
                )
                response.raise_for_status()

                # Try to parse JSON with better error handling
                try:
                    result_data = response.json()
                except json.JSONDecodeError as json_err:
                    logging.warning(f"JSON decode error for {cache_key}: {json_err}")
                    # Try to salvage partial JSON by truncating at last complete record
                    result_data = self._parse_partial_json(response.text, cache_key)
                    if result_data is None:
                        if attempt < retry_count - 1:
                            logging.info(
                                f"Retrying query {cache_key} due to JSON parsing issue"
                            )
                            time.sleep(2**attempt)  # Exponential backoff
                            continue
                        else:
                            logging.error(
                                f"Failed to parse JSON for {cache_key} after all retries"
                            )
                            return None

                result = result_data["results"]["bindings"]

                # Cache the successful result
                self.cache.set(cache_key, result, expire=30 * 86400)  # Cache for 30 days
                logging.debug(f"Cached result for {cache_key} ({len(result)} results)")

                return result

            except requests.RequestException as e:
                logging.warning(
                    f"Request failed for {cache_key} (attempt {attempt + 1}): {e}"
                )
                if attempt < retry_count - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logging.error(
                        f"SPARQL query failed for {cache_key} after all retries: {e}"
                    )
                    return None

        return None

    def _parse_partial_json(self, json_text: str, cache_key: str) -> Optional[Dict]:
        """Attempt to parse partial/truncated JSON response.

        Args:
            json_text: Raw JSON text that failed to parse.
            cache_key: Cache key for logging purposes.

        Returns:
            Parsed JSON dict or None if unsuccessful.
        """
        try:
            # Try to find the last complete binding entry
            lines = json_text.split("\n")

            # Look for the bindings array and try to reconstruct
            in_bindings = False
            bindings_start = -1

            for i, line in enumerate(lines):
                if '"bindings"' in line and "[" in line:
                    in_bindings = True
                    bindings_start = i
                    break

            if not in_bindings or bindings_start == -1:
                logging.error(
                    f"Could not find bindings array in partial JSON for {cache_key}"
                )
                return None

            # Try to reconstruct JSON by finding last complete entry
            for end_line in range(len(lines) - 1, bindings_start, -1):
                try:
                    # Try to close the JSON at this point
                    partial_lines = lines[: end_line + 1]

                    # Remove trailing commas and incomplete entries
                    last_line = partial_lines[-1].rstrip()
                    if last_line.endswith(","):
                        last_line = last_line[:-1]
                        partial_lines[-1] = last_line

                    # Close the JSON structure
                    partial_lines.extend(["]}}"])

                    reconstructed_json = "\n".join(partial_lines)
                    result = json.loads(reconstructed_json)

                    logging.info(
                        f"Successfully parsed partial JSON for {cache_key} "
                        f"({len(result.get('results', {}).get('bindings', []))} results)"
                    )
                    return result

                except json.JSONDecodeError:
                    continue

            logging.error(f"Could not reconstruct valid JSON for {cache_key}")
            return None

        except Exception as e:
            logging.error(f"Error parsing partial JSON for {cache_key}: {e}")
            return None

    def _build_total_entities_query(self, wikidata_id: str) -> str:
        """Build SPARQL query to count total entities within a domain.

        Args:
            wikidata_id: Wikidata ID of the domain (e.g., Q11344).

        Returns:
            SPARQL query string for counting total entities.
        """
        return f"""
        SELECT (COUNT(DISTINCT ?item) as ?total) WHERE {{
          ?item wdt:P31/wdt:P279* wd:{wikidata_id} .
        }}
        """

    def _build_bilingual_entities_query(
        self, wikidata_id: str, limit: int = 1000, offset: int = 0
    ) -> str:
        """Build SPARQL query to find entities with both EN and UK Wikipedia pages.

        Args:
            wikidata_id: Wikidata ID of the domain (e.g., Q11344).
            limit: Maximum number of results per query.
            offset: Offset for pagination.

        Returns:
            SPARQL query string for entities with both language versions.
        """
        return f"""
        SELECT ?item ?itemLabel ?enWiki ?ukWiki WHERE {{
          ?item wdt:P31/wdt:P279* wd:{wikidata_id} .
          ?enWiki schema:about ?item ; 
                 schema:isPartOf <https://en.wikipedia.org/> .
          ?ukWiki schema:about ?item ; 
                 schema:isPartOf <https://uk.wikipedia.org/> .
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en,uk" 
          }}
        }}
        LIMIT {limit}
        OFFSET {offset}
        """

    def _execute_paginated_query(
        self, wikidata_id: str, query_type: str = "bilingual"
    ) -> List[Dict]:
        """Execute a paginated SPARQL query to get all results.

        Args:
            wikidata_id: Wikidata ID for the domain.
            query_type: Type of query ("bilingual" only for now).

        Returns:
            Combined list of all results from paginated queries.
        """
        all_results = []
        offset = 0

        while True:
            cache_key = f"{query_type}_{wikidata_id}_page_{offset//self.page_size}"

            if query_type == "bilingual":
                query = self._build_bilingual_entities_query(
                    wikidata_id, self.page_size, offset
                )
            else:
                raise ValueError(f"Unknown query type: {query_type}")

            results = self._execute_sparql_query(query, cache_key)

            if not results:
                logging.warning(f"No results for {cache_key}")
                break

            all_results.extend(results)
            logging.info(
                f"Retrieved {len(results)} results for {cache_key} "
                f"(total so far: {len(all_results)})"
            )

            # If we got fewer results than the page size, we've reached the end
            if len(results) < self.page_size:
                break

            offset += self.page_size
            time.sleep(0.5)  # Small delay between pages to be respectful

        logging.info(
            f"Completed paginated query for {wikidata_id}: {len(all_results)} total results"
        )
        return all_results

    def _save_bilingual_entities_csv(
        self, domain_name: str, entities: List[Dict]
    ) -> Path:
        """Save bilingual entities to a CSV file.

        Args:
            domain_name: Name of the domain for the filename.
            entities: List of entity dictionaries from SPARQL results.

        Returns:
            Path to the saved CSV file.
        """
        # Create safe filename from domain name
        safe_name = "".join(
            c for c in domain_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_name = safe_name.replace(" ", "_").lower()
        csv_path = self.output_dir / f"{safe_name}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["wikidata_id", "english_wikipedia_page", "ukrainian_wikipedia_page"]
            )

            for entity in entities:
                wikidata_id = entity["item"]["value"].split("/")[
                    -1
                ]  # Extract Q-ID from URI
                en_page = entity["enWiki"]["value"]
                uk_page = entity["ukWiki"]["value"]
                writer.writerow([wikidata_id, en_page, uk_page])

        logging.info(f"Saved {len(entities)} bilingual entities to {csv_path}")
        return csv_path

    def validate_domain(self, domain_name: str, wikidata_id: str) -> DomainStats:
        """Validate a single domain's Wikipedia coverage.

        Args:
            domain_name: Human-readable domain name.
            wikidata_id: Wikidata ID for the domain.

        Returns:
            DomainStats with coverage information.
        """
        logging.info(f"Validating domain: {domain_name} ({wikidata_id})")

        # First query: get total count of entities in domain
        total_query = self._build_total_entities_query(wikidata_id)
        total_cache_key = f"total_{wikidata_id}"
        total_results = self._execute_sparql_query(total_query, total_cache_key)

        if not total_results:
            logging.warning(f"Could not get total count for domain {domain_name}")
            return DomainStats(domain_name, wikidata_id, 0, 0, 0.0)

        total_entities = int(total_results[0]["total"]["value"])

        # Second query: get entities with both EN and UK pages (paginated)
        logging.info(
            f"Fetching bilingual entities for {domain_name} using pagination..."
        )
        bilingual_results = self._execute_paginated_query(wikidata_id, "bilingual")

        if not bilingual_results:
            both_languages = 0
            logging.warning(f"No bilingual entities found for {domain_name}")
        else:
            both_languages = len(bilingual_results)
            # Save bilingual entities to CSV
            try:
                csv_path = self._save_bilingual_entities_csv(
                    domain_name, bilingual_results
                )
                logging.info(
                    f"Saved bilingual entities for {domain_name} to {csv_path}"
                )
            except Exception as e:
                logging.error(f"Failed to save CSV for {domain_name}: {e}")

        # Calculate coverage percentage
        coverage_percent = (
            (both_languages / total_entities * 100) if total_entities > 0 else 0.0
        )

        logging.info(
            f"Found {total_entities} total entities, "
            f"{both_languages} with both languages ({coverage_percent:.1f}%)"
        )

        return DomainStats(
            domain_name, wikidata_id, total_entities, both_languages, coverage_percent
        )

    def validate_domains_from_csv(self, csv_path: Path) -> List[DomainStats]:
        """Validate all domains from a CSV file.

        Args:
            csv_path: Path to CSV file with columns: domain_name, wikidata_id.

        Returns:
            List of DomainStats for all domains.
        """
        results = []

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i > 0:  # Add delay between requests to be respectful
                    time.sleep(1)

                stats = self.validate_domain(row["domain_name"], row["wikidata_id"])
                results.append(stats)

        return results


def print_results_table(stats: List[DomainStats]) -> None:
    """Print validation results as a formatted table.

    Args:
        stats: List of domain statistics to display.
    """
    print("\n" + "=" * 80)
    print("BILINGUAL DOMAIN VALIDATION RESULTS")
    print("=" * 80)

    header = f"{'Domain':<25} {'ID':<10} {'Total':<7} {'Both':<6} {'Coverage':<10}"
    print(header)
    print("-" * len(header))

    total_entities = 0
    total_both = 0

    for stat in stats:
        print(
            f"{stat.domain_name:<25} {stat.wikidata_id:<10} "
            f"{stat.total_entities:<7} {stat.both_languages:<6} "
            f"{stat.coverage_percent:>7.1f}%"
        )

        total_entities += stat.total_entities
        total_both += stat.both_languages

    print("-" * len(header))
    overall_coverage = (
        (total_both / total_entities * 100) if total_entities > 0 else 0.0
    )
    print(
        f"{'TOTAL':<25} {'':<10} {total_entities:<7} {total_both:<6} "
        f"{overall_coverage:>7.1f}%"
    )

    print(f"\nSummary:")
    print(f"- Total entities across all domains: {total_entities:,}")
    print(f"- Entities with both EN/UK pages: {total_both:,} ({overall_coverage:.1f}%)")
    print(f"- Average entities per domain: {total_entities/len(stats):.1f}")
    print(f"- Average bilingual entities per domain: {total_both/len(stats):.1f}")
    print(f"- Individual domain CSV files created in domain_entities/ directory")

    # Show top domains by absolute count and by percentage
    stats_sorted_by_count = sorted(stats, key=lambda x: x.both_languages, reverse=True)[
        :5
    ]
    stats_sorted_by_coverage = sorted(
        stats, key=lambda x: x.coverage_percent, reverse=True
    )[:5]

    print(f"\nTop 5 domains by bilingual entity count:")
    for i, stat in enumerate(stats_sorted_by_count, 1):
        print(
            f"  {i}. {stat.domain_name}: {stat.both_languages} entities ({stat.coverage_percent:.1f}%)"
        )

    print(f"\nTop 5 domains by coverage percentage:")
    for i, stat in enumerate(stats_sorted_by_coverage, 1):
        print(
            f"  {i}. {stat.domain_name}: {stat.coverage_percent:.1f}% ({stat.both_languages} entities)"
        )


def main() -> None:
    """Main entry point for the domain validation script."""
    parser = argparse.ArgumentParser(
        description="Validate bilingual thesaurus domains using Wikidata. "
        "Creates CSV files for each domain with bilingual entities."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV file containing domain_name,wikidata_id columns",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for caching SPARQL results (default: .cache)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Number of results per paginated query (default: 1000)",
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

    if not args.csv_file.exists():
        logging.error(f"CSV file not found: {args.csv_file}")
        sys.exit(1)

    validator = WikidataValidator(cache_dir=args.cache_dir, page_size=args.page_size)

    try:
        stats = validator.validate_domains_from_csv(args.csv_file)
        print_results_table(stats)
        print(f"\nBilingual entity CSV files saved to: {validator.output_dir}/")
    except KeyboardInterrupt:
        logging.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
