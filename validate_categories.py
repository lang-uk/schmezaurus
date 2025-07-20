import argparse
import csv
import hashlib
import json
import logging
import sys
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import diskcache as dc
import requests
import wikipediaapi
from tqdm import tqdm


@dataclass
class ProcessingStats:
    """Container for processing statistics with comprehensive metrics."""

    total_categories: int = 0
    total_pages_found: int = 0
    pages_with_wikidata: int = 0
    pages_with_english_mapping: int = 0
    failed_pages: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    error_summary: Dict[str, int] = field(default_factory=dict)

    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        return (
            self.end_time - self.start_time
            if self.end_time > 0
            else time.time() - self.start_time
        )

    @property
    def wikidata_success_rate(self) -> float:
        """Calculate Wikidata ID extraction success rate."""
        if self.total_pages_found == 0:
            return 0.0
        return (self.pages_with_wikidata / self.total_pages_found) * 100

    @property
    def mapping_success_rate(self) -> float:
        """Calculate English mapping success rate."""
        if self.pages_with_wikidata == 0:
            return 0.0
        return (self.pages_with_english_mapping / self.pages_with_wikidata) * 100

    def record_error(self, error_type: str) -> None:
        """Record an error in the summary."""
        self.error_summary[error_type] = self.error_summary.get(error_type, 0) + 1

    def finalize(self) -> None:
        """Finalize statistics calculation."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "total_categories": self.total_categories,
            "total_pages_found": self.total_pages_found,
            "pages_with_wikidata": self.pages_with_wikidata,
            "pages_with_english_mapping": self.pages_with_english_mapping,
            "failed_pages": self.failed_pages,
            "wikidata_success_rate_pct": round(self.wikidata_success_rate, 2),
            "mapping_success_rate_pct": round(self.mapping_success_rate, 2),
            "processing_time_sec": round(self.processing_time, 2),
            "error_summary": self.error_summary,
        }


class WikipediaExtractor:
    """Main Wikipedia extractor with comprehensive functionality."""

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
            user_agent: User agent string for API requests.
            bearer_token: Optional bearer token for authentication.
            rate_limit: Minimum seconds between requests.
        """
        self.cache = dc.Cache(str(cache_dir / "wikipedia_api"))
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.stats = ProcessingStats()
        self.logger = logging.getLogger(__name__)

        # Prepare headers for bearer token authentication
        headers = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
            self.logger.info("Using bearer token authentication")
        else:
            self.logger.info("Using unauthenticated requests")

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

        # Session for Wikidata API calls with optimization
        self.session = self._create_optimized_session()

    def _create_optimized_session(self) -> requests.Session:
        """Create optimized requests session for Wikidata API."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "WikipediaExtractor/1.0 (contact@example.com)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }
        )

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=50, max_retries=0
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    @contextmanager
    def _rate_limit_context(self):
        """Context manager for rate limiting API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            time.sleep(sleep_time)

        yield

        self.last_request_time = time.time()

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache keys for API requests."""
        sorted_params = sorted(params.items()) if params else []
        key_data = f"{endpoint}:{json.dumps(sorted_params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get_category_pages(
        self,
        category_titles: List[str],
        max_depth: int = 2,
        max_pages: Optional[int] = None,
    ) -> List[str]:
        """Recursively extract all pages from given categories.

        Args:
            category_titles: List of category titles to process.
            max_depth: Maximum recursion depth for subcategories.
            max_pages: Maximum number of pages to extract (None for unlimited).

        Returns:
            List of unique page titles found in categories.
        """
        all_pages: Set[str] = set()
        visited_categories: Set[str] = set()
        self.stats.total_categories = len(category_titles)

        self.logger.info(
            f"Processing {len(category_titles)} categories with max depth {max_depth}"
        )

        with tqdm(category_titles, desc="Processing categories", unit="cat") as pbar:
            for category_title in pbar:
                try:
                    category_pages = self._extract_category_recursive(
                        category_title, max_depth, visited_categories, max_pages
                    )
                    all_pages.update(category_pages)

                    pbar.set_postfix(
                        {
                            "Total Pages": len(all_pages),
                            "Categories Visited": len(visited_categories),
                        }
                    )

                    # Stop if we've reached max_pages
                    if max_pages and len(all_pages) >= max_pages:
                        self.logger.info(f"Reached maximum pages limit: {max_pages}")
                        break

                except Exception as e:
                    self.logger.error(
                        f"Error processing category '{category_title}': {e}"
                    )
                    self.stats.record_error(f"category_error_{type(e).__name__}")

        self.stats.total_pages_found = len(all_pages)
        self.logger.info(
            f"Found {len(all_pages)} unique pages across {len(visited_categories)} categories"
        )

        return list(all_pages)[:max_pages] if max_pages else list(all_pages)

    def _extract_category_recursive(
        self,
        category_title: str,
        max_depth: int,
        visited: Set[str],
        max_pages: Optional[int],
        current_depth: int = 0,
    ) -> Set[str]:
        """Recursively extract pages from a category and its subcategories."""
        if current_depth > max_depth or category_title in visited:
            return set()

        visited.add(category_title)
        pages: Set[str] = set()

        try:
            with self._rate_limit_context():
                category = self.wiki_uk.page(category_title)

            if not category.exists():
                self.logger.warning(f"Category does not exist: {category_title}")
                return set()

            # Extract pages and subcategories
            for title, page in category.categorymembers.items():
                if max_pages and len(pages) >= max_pages:
                    break

                if page.ns == wikipediaapi.Namespace.MAIN:
                    # Regular article
                    pages.add(title)
                elif (
                    page.ns == wikipediaapi.Namespace.CATEGORY
                    and current_depth < max_depth
                ):
                    # Subcategory - recurse
                    subcategory_pages = self._extract_category_recursive(
                        title, max_depth, visited, max_pages, current_depth + 1
                    )
                    pages.update(subcategory_pages)

        except Exception as e:
            self.logger.error(f"Error processing category '{category_title}': {e}")
            self.stats.record_error(f"category_extraction_error_{type(e).__name__}")

        return pages

    def get_wikidata_id(self, page_title: str, lang: str = "uk") -> Optional[str]:
        """Extract Wikidata ID from Wikipedia page title.

        Args:
            page_title: Wikipedia page title.
            lang: Language code (default: uk).

        Returns:
            Wikidata ID (e.g., 'Q12345') or None if not found.
        """
        cache_key = self._generate_cache_key(
            "wikidata_id", {"title": page_title, "lang": lang}
        )

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "pageprops",
            "format": "json",
            "titles": page_title,
            "redirects": 1,
        }

        try:
            with self._rate_limit_context():
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":  # Page exists
                    pageprops = page_data.get("pageprops", {})
                    wikidata_id = pageprops.get("wikibase_item")

                    # Cache result (including None results to avoid repeated requests)
                    self.cache.set(cache_key, wikidata_id, expire=3600 * 24)
                    return wikidata_id

            # Cache None result
            self.cache.set(cache_key, None, expire=3600 * 6)
            return None

        except Exception as e:
            self.logger.error(f"Error fetching Wikidata ID for '{page_title}': {e}")
            self.stats.record_error(f"wikidata_fetch_error_{type(e).__name__}")
            return None

    def get_english_page_title(self, wikidata_id: str) -> Optional[str]:
        """Get English Wikipedia page title from Wikidata ID.

        Args:
            wikidata_id: Wikidata ID (e.g., 'Q12345').

        Returns:
            English Wikipedia page title or None if not found.
        """
        cache_key = self._generate_cache_key(
            "english_title", {"wikidata_id": wikidata_id}
        )

        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": wikidata_id,
            "props": "sitelinks/urls",
            "sitefilter": "enwiki",
        }

        try:
            with self._rate_limit_context():
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

            entities = data.get("entities", {})
            entity = entities.get(wikidata_id, {})
            sitelinks = entity.get("sitelinks", {})

            enwiki_data = sitelinks.get("enwiki")
            if enwiki_data:
                english_title = enwiki_data.get("title")
                # Cache successful result
                self.cache.set(cache_key, english_title, expire=3600 * 24)
                return english_title

            # Cache None result
            self.cache.set(cache_key, None, expire=3600 * 6)
            return None

        except Exception as e:
            self.logger.error(
                f"Error fetching English title for Wikidata ID '{wikidata_id}': {e}"
            )
            self.stats.record_error(f"english_fetch_error_{type(e).__name__}")
            return None

    def process_pages(self, page_titles: List[str]) -> List[Tuple[str, str, str]]:
        """Process Ukrainian Wikipedia pages to find English mappings.

        Args:
            page_titles: List of Ukrainian Wikipedia page titles.

        Returns:
            List of tuples (wikidata_id, english_title, ukrainian_title).
        """
        results: List[Tuple[str, str, str]] = []

        self.logger.info(
            f"Processing {len(page_titles)} pages for cross-language mapping"
        )

        with tqdm(page_titles, desc="Processing pages", unit="page") as pbar:
            for uk_title in pbar:
                try:
                    # Get Wikidata ID
                    wikidata_id = self.get_wikidata_id(uk_title, "uk")

                    if wikidata_id:
                        self.stats.pages_with_wikidata += 1

                        # Get English page title
                        en_title = self.get_english_page_title(wikidata_id)

                        if en_title:
                            self.stats.pages_with_english_mapping += 1
                            results.append(
                                (
                                    wikidata_id,
                                    f"https://en.wikipedia.org/wiki/{en_title}",
                                    f"https://uk.wikipedia.org/wiki/{uk_title}",
                                )
                            )
                        else:
                            self.logger.debug(
                                f"No English mapping for '{uk_title}' (Wikidata: {wikidata_id})"
                            )
                            self.stats.record_error("no_english_mapping")
                    else:
                        self.logger.debug(f"No Wikidata ID found for '{uk_title}'")
                        self.stats.record_error("no_wikidata_id")
                        self.stats.failed_pages += 1

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "Wikidata IDs": self.stats.pages_with_wikidata,
                            "English Mapped": self.stats.pages_with_english_mapping,
                            "Success Rate": f"{self.stats.mapping_success_rate:.1f}%",
                        }
                    )

                except Exception as e:
                    self.logger.error(f"Error processing page '{uk_title}': {e}")
                    self.stats.record_error(f"processing_error_{type(e).__name__}")
                    self.stats.failed_pages += 1

        self.logger.info(
            f"Successfully mapped {len(results)} pages to English Wikipedia"
        )
        return results

    def write_csv_output(
        self, results: List[Tuple[str, str, str]], output_path: Path
    ) -> None:
        """Write results to CSV file.

        Args:
            results: List of (wikidata_id, english_title, ukrainian_title) tuples.
            output_path: Path to output CSV file.
        """
        self.logger.info(f"Writing {len(results)} results to {output_path}")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

                # Write header
                writer.writerow(
                    [
                        "wikidata_id",
                        "english_wikipedia_page",
                        "ukrainian_wikipedia_page",
                    ]
                )

                # Write data
                for wikidata_id, en_title, uk_title in results:
                    writer.writerow([wikidata_id, en_title, uk_title])

            self.logger.info(f"Successfully wrote CSV output to {output_path}")

        except Exception as e:
            self.logger.error(f"Error writing CSV output: {e}")
            raise

    def get_statistics_report(self) -> str:
        """Generate comprehensive statistics report.

        Returns:
            Formatted statistics report string.
        """
        stats_dict = self.stats.to_dict()

        report = f"""
Wikipedia Mining Statistics Report
==================================
Processing Summary:
  - Categories processed: {stats_dict['total_categories']}
  - Total pages found: {stats_dict['total_pages_found']:,}
  - Pages with Wikidata IDs: {stats_dict['pages_with_wikidata']:,}
  - Pages mapped to English: {stats_dict['pages_with_english_mapping']:,}
  - Failed pages: {stats_dict['failed_pages']:,}

Success Rates:
  - Wikidata extraction: {stats_dict['wikidata_success_rate_pct']:.1f}%
  - English mapping: {stats_dict['mapping_success_rate_pct']:.1f}%

Performance:
  - Total processing time: {stats_dict['processing_time_sec']:.2f} seconds
  - Processing rate: {stats_dict['total_pages_found'] / max(1, stats_dict['processing_time_sec']):.2f} pages/second

Error Summary:"""

        if stats_dict["error_summary"]:
            for error_type, count in stats_dict["error_summary"].items():
                report += f"\n  - {error_type}: {count:,} occurrences"
        else:
            report += "\n  - No errors encountered"

        return report


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Mine Ukrainian Wikipedia categories and map to English pages via Wikidata.",
        epilog='Example: python wikipedia_miner.py "Category:Physics" --output results.csv --max-pages 1000',
    )

    # Required arguments
    parser.add_argument(
        "categories", nargs="+", help="Ukrainian Wikipedia category titles to process"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wikipedia_mining_results.csv"),
        help="Output CSV file path (default: wikipedia_mining_results.csv)",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache directory for API responses (default: .cache)",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum recursion depth for subcategories (default: 2)",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process (default: unlimited)",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Minimum seconds between API requests (default: 0.1)",
    )

    parser.add_argument(
        "--bearer-token", type=str, help="Bearer token for Wikipedia API authentication"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument("--log-file", type=Path, help="Optional log file path")

    parser.add_argument(
        "--user-agent",
        type=str,
        default="WikipediaExtractor/1.0 (Educational Research Tool)",
        help="User agent string for API requests",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate parsed command line arguments.

    Args:
        args: Parsed command line arguments.

    Raises:
        ValueError: If validation fails.
    """
    if args.max_depth < 0:
        raise ValueError("max-depth must be non-negative")

    if args.max_pages is not None and args.max_pages <= 0:
        raise ValueError("max-pages must be positive")

    if args.rate_limit < 0:
        raise ValueError("rate-limit must be non-negative")

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Validate category titles format
    for category in args.categories:
        if not category.startswith("Category:"):
            raise ValueError(f"Category titles must start with 'Category:': {category}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main application function.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Setup logging first
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    try:
        # Validate arguments
        validate_arguments(args)

        logger.info("Starting Wikipedia mining process")
        logger.info(f"Categories: {args.categories}")
        logger.info(f"Output file: {args.output}")
        logger.info(
            f"Max depth: {args.max_depth}, Max pages: {args.max_pages or 'unlimited'}"
        )

        # Initialize extractor
        extractor = WikipediaExtractor(
            cache_dir=args.cache_dir,
            user_agent=args.user_agent,
            bearer_token=args.bearer_token,
            rate_limit=args.rate_limit,
        )

        # Extract pages from categories
        logger.info("Step 1: Extracting pages from categories")
        page_titles = extractor.get_category_pages(
            args.categories, max_depth=args.max_depth, max_pages=args.max_pages
        )

        if not page_titles:
            logger.warning("No pages found in specified categories")
            return 1

        # Process pages for cross-language mapping
        logger.info("Step 2: Processing pages for cross-language mapping")
        results = extractor.process_pages(page_titles)

        if not results:
            logger.warning("No successful cross-language mappings found")
            return 1

        # Write CSV output
        logger.info("Step 3: Writing CSV output")
        extractor.write_csv_output(results, args.output)

        # Generate and display statistics
        extractor.stats.finalize()
        stats_report = extractor.get_statistics_report()
        print(stats_report)

        logger.info("Wikipedia mining completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Application failed: {e}")
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
