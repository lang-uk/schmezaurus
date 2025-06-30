#!/usr/bin/env python3
"""Extract Wikipedia content using the wikipedia-api library with bearer token support.

This script reads a CSV file containing Wikidata IDs and Wikipedia URLs,
uses the wikipedia-api library with optional bearer token authentication
to extract clean text content, and saves results as JSONL files.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import diskcache as dc
import wikipediaapi


class WikipediaAPIExtractor:
    """Extracts Wikipedia content using the wikipedia-api library."""

    def __init__(
        self,
        cache_dir: Path = Path(".cache"),
        user_agent: str = "WikipediaExtractor/1.0 (Educational Research Tool)",
        bearer_token: Optional[str] = None,
        rate_limit: float = 0.1,
    ) -> None:
        """Initialize the Wikipedia API extractor.

        Args:
            cache_dir: Directory for caching API responses.
            user_agent: User agent string for Wikipedia API requests.
            bearer_token: Optional bearer token for authentication.
            rate_limit: Minimum seconds between API requests.
        """
        self.cache = dc.Cache(str(cache_dir / "wikipedia_api"))
        self.rate_limit = rate_limit
        self.last_request_time = 0.0

        # Prepare headers for bearer token authentication
        headers = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
            logging.info("Using bearer token authentication")
        else:
            logging.info("Using unauthenticated requests")

        # Initialize Wikipedia API clients for both languages
        self.wiki_en = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            headers=headers if headers else None,
        )

        self.wiki_uk = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language="uk",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            headers=headers if headers else None,
        )

        self.output_dir = Path("extracted_texts")
        self.output_dir.mkdir(exist_ok=True)

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _extract_page_title_from_url(self, wikipedia_url: str) -> Optional[str]:
        """Extract page title from Wikipedia URL.

        Args:
            wikipedia_url: Full Wikipedia URL.

        Returns:
            Page title or None if URL is invalid.
        """
        try:
            parsed_url = urlparse(wikipedia_url)
            if "wiki/" not in parsed_url.path:
                logging.warning(f"Invalid Wikipedia URL format: {wikipedia_url}")
                return None

            # Extract title after /wiki/
            title_part = parsed_url.path.split("/wiki/")[-1]
            # URL decode the title
            title = unquote(title_part)

            # No need to replace underscores - wikipedia-api handles this
            return title

        except Exception as e:
            logging.error(f"Failed to extract title from URL {wikipedia_url}: {e}")
            return None

    def _get_wikipedia_page_content(
        self, page_title: str, language: str
    ) -> Optional[Dict[str, str]]:
        """Get clean text content from Wikipedia page using wikipedia-api.

        Args:
            page_title: Wikipedia page title.
            language: Language code ('en' or 'uk').

        Returns:
            Dictionary with page content or None if failed.
        """
        cache_key = f"wikipediaapi_{language}_{page_title}"

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            logging.debug(f"Using cached result for {language}:{page_title}")
            return cached_result

        self._enforce_rate_limit()

        try:
            wiki_client = self.wiki_en if language == "en" else self.wiki_uk

            logging.debug(f"Fetching {language}:{page_title}")
            page = wiki_client.page(page_title)

            if not page.exists():
                logging.warning(f"Page does not exist: {language}:{page_title}")
                result = None
            else:
                # Extract comprehensive page information using wikipedia-api
                result = {
                    "title": page.title,
                    "text": page.text,  # Full text including sections
                    "summary": page.summary,
                    "url": page.fullurl,
                    "pageid": page.pageid,
                    "language": language,
                    "sections": [section.title for section in page.sections],
                    "source": "wikipedia_api",
                }

                # Basic quality checks
                if len(result["text"]) < 100:
                    logging.warning(
                        f"Very short article: {language}:{page_title} ({len(result['text'])} chars)"
                    )

            # Cache the result (including None for non-existent pages)
            self.cache.set(cache_key, result, expire=604800)  # Cache for 7 days
            logging.debug(f"Cached result for {language}:{page_title}")

            return result

        except Exception as e:
            logging.error(f"Failed to fetch {language}:{page_title}: {e}")
            return None

    def process_csv_to_jsonl(self, csv_path: Path) -> Tuple[int, int, int]:
        """Process CSV file and create JSONL files for both languages.

        Args:
            csv_path: Path to input CSV file.

        Returns:
            Tuple of (successful_extractions, total_entries, failed_extractions).
        """
        domain_name = csv_path.stem

        en_jsonl_path = self.output_dir / f"{domain_name}_en.jsonl"
        uk_jsonl_path = self.output_dir / f"{domain_name}_uk.jsonl"

        successful_extractions = 0
        failed_extractions = 0
        total_entries = 0

        with (
            csv_path.open("r", encoding="utf-8") as csv_file,
            en_jsonl_path.open("w", encoding="utf-8") as en_file,
            uk_jsonl_path.open("w", encoding="utf-8") as uk_file,
        ):
            reader = csv.DictReader(csv_file)

            for row in reader:
                total_entries += 1
                wikidata_id = row["wikidata_id"]
                en_url = row["english_wikipedia_page"]
                uk_url = row["ukrainian_wikipedia_page"]

                logging.info(f"Processing {wikidata_id} ({total_entries})")

                # Extract page titles from URLs
                en_title = self._extract_page_title_from_url(en_url)
                uk_title = self._extract_page_title_from_url(uk_url)

                entry_success = False

                # Process English page
                if en_title:
                    en_content = self._get_wikipedia_page_content(en_title, "en")
                    if en_content:
                        en_record = {
                            "wikidata_id": wikidata_id,
                            "original_url": en_url,
                            **en_content,
                        }
                        en_file.write(json.dumps(en_record, ensure_ascii=False) + "\n")
                        entry_success = True
                    else:
                        logging.warning(
                            f"Failed to extract English content for {wikidata_id}"
                        )
                else:
                    logging.error(f"Could not extract English title from URL: {en_url}")

                # Process Ukrainian page
                if uk_title:
                    uk_content = self._get_wikipedia_page_content(uk_title, "uk")
                    if uk_content:
                        uk_record = {
                            "wikidata_id": wikidata_id,
                            "original_url": uk_url,
                            **uk_content,
                        }
                        uk_file.write(json.dumps(uk_record, ensure_ascii=False) + "\n")
                        entry_success = True
                    else:
                        logging.warning(
                            f"Failed to extract Ukrainian content for {wikidata_id}"
                        )
                else:
                    logging.error(
                        f"Could not extract Ukrainian title from URL: {uk_url}"
                    )

                # Count results
                if entry_success:
                    successful_extractions += 1
                else:
                    failed_extractions += 1

        logging.info(
            f"Extraction complete: {successful_extractions}/{total_entries} successful, "
            f"{failed_extractions} failed"
        )
        logging.info(f"Output files: {en_jsonl_path}, {uk_jsonl_path}")

        return successful_extractions, total_entries, failed_extractions


def main() -> None:
    """Main entry point for Wikipedia API text extraction."""
    parser = argparse.ArgumentParser(
        description="Extract Wikipedia content using the wikipedia-api library. "
        "Supports bearer token authentication via headers parameter. "
        "Processes CSV files with wikidata_id, english_wikipedia_page, "
        "ukrainian_wikipedia_page columns and creates clean JSONL files."
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file containing Wikipedia URLs"
    )
    parser.add_argument(
        "--bearer-token", type=str, help="Bearer token for Wikipedia API authentication"
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="WikipediaExtractor/1.0 (Educational Research Tool)",
        help="User agent string for Wikipedia API requests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_texts"),
        help="Directory for output JSONL files (default: extracted_texts)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for caching API responses (default: .cache)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Minimum seconds between API requests (default: 0.1)",
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
        return

    extractor = WikipediaAPIExtractor(
        cache_dir=args.cache_dir,
        user_agent=args.user_agent,
        bearer_token=args.bearer_token,
        rate_limit=args.rate_limit,
    )

    # Override output directory if specified
    if args.output_dir != Path("extracted_texts"):
        extractor.output_dir = args.output_dir
        extractor.output_dir.mkdir(exist_ok=True)

    try:
        successful, total, failed = extractor.process_csv_to_jsonl(args.csv_file)

        print(f"\n{'='*70}")
        print("WIKIPEDIA-API EXTRACTION RESULTS")
        print(f"{'='*70}")
        print(f"Input file: {args.csv_file}")
        print(f"Total entries processed: {total}")
        print(f"Successful extractions: {successful}")
        print(f"Failed extractions: {failed}")
        print(f"Success rate: {successful/total*100:.1f}%")
        print(f"Output directory: {extractor.output_dir}")
        print(f"JSONL files created:")
        print(f"  - {args.csv_file.stem}_en.jsonl")
        print(f"  - {args.csv_file.stem}_uk.jsonl")
        print(f"\nAPI Details:")
        print(f"  - Library: wikipedia-api (martin-majlis/Wikipedia-API)")
        print(
            f"  - Authentication: {'Bearer token (via headers)' if args.bearer_token else 'User-agent only'}"
        )
        print(f"  - Extract format: WIKI (clean text)")
        print(f"  - Rate limit: {args.rate_limit}s between requests")
        print(f"  - User agent: {args.user_agent}")
        print(f"  - Cache directory: {args.cache_dir}")

        if not args.bearer_token:
            print(f"\nTip: You can use bearer token authentication with --bearer-token")
            print(f"     This may provide better rate limits and reliability")

    except KeyboardInterrupt:
        logging.info("Extraction interrupted by user")
    except Exception as e:
        logging.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
