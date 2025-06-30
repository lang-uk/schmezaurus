#!/usr/bin/env python3
"""Visualize term embeddings using t-SNE in 2D or 3D interactive plots.

This script loads multiple embedding files, applies t-SNE dimensionality reduction,
and creates an interactive visualization to explore how terms from different
languages/domains cluster in the embedding space. Supports both 2D and 3D
visualizations for comprehensive analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class EmbeddingVisualizer:
    """Visualizes term embeddings using t-SNE and interactive plots."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the embedding visualizer.

        Args:
            random_state: Random state for reproducible t-SNE results.
        """
        self.random_state = random_state
        self.data = []
        self.embeddings = None
        self.tsne_results = None

    def load_embeddings_from_jsonl(
        self,
        jsonl_path: Path,
        max_terms: Optional[int] = None,
        source_label: Optional[str] = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        """Load embeddings from a JSONL file.

        Args:
            jsonl_path: Path to JSONL embedding file.
            max_terms: Maximum number of terms to load (None for all).
            source_label: Label for this source (uses filename if None).

        Returns:
            Tuple of (term_data_list, embeddings_array).
        """
        if source_label is None:
            source_label = jsonl_path.stem

        term_data = []
        embeddings = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())

                    # Skip metadata lines
                    if record.get("type") == "metadata":
                        continue

                    # Extract required fields
                    term = record["term"]
                    score = record["score"]
                    embedding = np.array(record["embedding"])

                    term_data.append(
                        {
                            "term": term,
                            "score": score,
                            "source": source_label,
                            "file": jsonl_path.name,
                        }
                    )
                    embeddings.append(embedding)

                    # Stop if we've reached the limit
                    if max_terms and len(term_data) >= max_terms:
                        break

                except (json.JSONDecodeError, KeyError) as e:
                    logging.warning(
                        f"Skipping invalid line {line_num} in {jsonl_path}: {e}"
                    )

        embeddings_array = np.array(embeddings) if embeddings else np.empty((0, 0))
        logging.info(f"Loaded {len(term_data)} terms from {jsonl_path}")

        return term_data, embeddings_array

    def load_multiple_files(
        self,
        file_paths: List[Path],
        max_terms_per_file: Optional[int] = None,
        source_labels: Optional[List[str]] = None,
    ) -> None:
        """Load embeddings from multiple files.

        Args:
            file_paths: List of paths to JSONL embedding files.
            max_terms_per_file: Maximum terms to load per file.
            source_labels: Custom labels for each source file.
        """
        all_term_data = []
        all_embeddings = []

        if source_labels is None:
            source_labels = [None] * len(file_paths)
        elif len(source_labels) != len(file_paths):
            raise ValueError("Number of source labels must match number of files")

        for file_path, source_label in zip(file_paths, source_labels):
            term_data, embeddings = self.load_embeddings_from_jsonl(
                file_path, max_terms_per_file, source_label
            )

            if len(term_data) > 0:
                all_term_data.extend(term_data)
                all_embeddings.append(embeddings)
            else:
                logging.warning(f"No valid embeddings found in {file_path}")

        if not all_embeddings:
            raise ValueError("No valid embeddings found in any file")

        # Combine all embeddings
        self.data = all_term_data
        self.embeddings = np.vstack(all_embeddings)

        logging.info(
            f"Total loaded: {len(self.data)} terms with {self.embeddings.shape[1]}-dim embeddings"
        )

    def apply_tsne(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        learning_rate: str = "auto",
        normalize: bool = True,
    ) -> np.ndarray:
        """Apply t-SNE dimensionality reduction to embeddings.

        Args:
            n_components: Number of dimensions for t-SNE (2 or 3).
            perplexity: t-SNE perplexity parameter.
            max_iter: Maximum number of iterations for optimization.
            learning_rate: Learning rate for t-SNE.
            normalize: Whether to normalize embeddings before t-SNE.

        Returns:
            t-SNE coordinates with shape (n_samples, n_components).
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_multiple_files first.")

        if n_components not in [2, 3]:
            raise ValueError("n_components must be 2 or 3")

        logging.info(
            f"Applying {n_components}D t-SNE to {self.embeddings.shape[0]} embeddings..."
        )

        # Optionally normalize embeddings
        embeddings_to_use = self.embeddings
        if normalize:
            scaler = StandardScaler()
            embeddings_to_use = scaler.fit_transform(self.embeddings)
            logging.info("Normalized embeddings before t-SNE")

        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=self.random_state,
            verbose=1,
        )

        self.tsne_results = tsne.fit_transform(embeddings_to_use)
        logging.info(f"t-SNE completed. Final KL divergence: {tsne.kl_divergence_:.4f}")

        return self.tsne_results

    def create_interactive_plot(
        self,
        output_path: Optional[Path] = None,
        title: str = "Term Embeddings Visualization",
        width: int = 1200,
        height: int = 800,
        point_size: int = 8,
    ) -> go.Figure:
        """Create an interactive plotly visualization for 2D or 3D embeddings.

        Args:
            output_path: Path to save HTML file (None for display only).
            title: Plot title.
            width: Plot width in pixels.
            height: Plot height in pixels.
            point_size: Size of scatter plot points.

        Returns:
            Plotly figure object.
        """
        if self.tsne_results is None:
            raise ValueError("No t-SNE results. Call apply_tsne first.")

        n_dimensions = self.tsne_results.shape[1]

        # Create DataFrame for plotly
        df_data = {
            "term": [item["term"] for item in self.data],
            "score": [item["score"] for item in self.data],
            "source": [item["source"] for item in self.data],
            "file": [item["file"] for item in self.data],
        }

        # Add coordinate columns based on dimensionality
        if n_dimensions == 2:
            df_data.update({"x": self.tsne_results[:, 0], "y": self.tsne_results[:, 1]})
        elif n_dimensions == 3:
            df_data.update(
                {
                    "x": self.tsne_results[:, 0],
                    "y": self.tsne_results[:, 1],
                    "z": self.tsne_results[:, 2],
                }
            )
        else:
            raise ValueError(f"Unsupported dimensionality: {n_dimensions}")

        df = pd.DataFrame(df_data)

        # Create color mapping for sources
        unique_sources = df["source"].unique()
        colors = px.colors.qualitative.Set1[: len(unique_sources)]
        color_map = dict(zip(unique_sources, colors))

        # Create the appropriate scatter plot
        if n_dimensions == 2:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="source",
                hover_data=["term", "score", "file"],
                title=title,
                color_discrete_map=color_map,
                width=width,
                height=height,
            )

            # Update hover template for 2D
            hover_template = (
                "<b>%{customdata[0]}</b><br>"
                + "Score: %{customdata[1]:.4f}<br>"
                + "Source: %{fullData.name}<br>"
                + "File: %{customdata[2]}<br>"
                + "Position: (%{x:.2f}, %{y:.2f})<br>"
                + "<extra></extra>"
            )

            # Update layout for 2D
            fig.update_layout(
                xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2"
            )

        else:  # 3D case
            fig = px.scatter_3d(
                df,
                x="x",
                y="y",
                z="z",
                color="source",
                hover_data=["term", "score", "file"],
                title=title,
                color_discrete_map=color_map,
                width=width,
                height=height,
            )

            # Update hover template for 3D
            hover_template = (
                "<b>%{customdata[0]}</b><br>"
                + "Score: %{customdata[1]:.4f}<br>"
                + "Source: %{fullData.name}<br>"
                + "File: %{customdata[2]}<br>"
                + "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
                + "<extra></extra>"
            )

            # Update layout for 3D
            fig.update_layout(
                scene=dict(
                    xaxis_title="t-SNE Dimension 1",
                    yaxis_title="t-SNE Dimension 2",
                    zaxis_title="t-SNE Dimension 3",
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5),
                    ),
                )
            )

        # Update traces for better visualization
        fig.update_traces(
            marker=dict(
                size=point_size, opacity=0.7, line=dict(width=1, color="white")
            ),
            hovertemplate=hover_template,
        )

        # Update common layout elements
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
            margin=dict(r=150),  # Make room for legend
        )

        # Add source statistics as annotation
        stats_text = self._create_stats_text(df, n_dimensions)
        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

        # Save to HTML if path provided
        if output_path:
            fig.write_html(str(output_path))
            logging.info(f"Interactive {n_dimensions}D plot saved to {output_path}")

        return fig

    def _create_stats_text(self, df: pd.DataFrame, n_dimensions: int) -> str:
        """Create statistics text for the plot annotation.

        Args:
            df: DataFrame with plot data.
            n_dimensions: Number of dimensions (2 or 3).

        Returns:
            Formatted statistics string.
        """
        stats = []
        stats.append("<b>Dataset Statistics:</b>")
        stats.append(f"Total terms: {len(df)}")
        stats.append(f"Dimensions: {n_dimensions}D t-SNE")
        stats.append("")

        for source in df["source"].unique():
            source_df = df[df["source"] == source]
            avg_score = source_df["score"].mean()
            stats.append(
                f"<b>{source}:</b> {len(source_df)} terms (avg score: {avg_score:.3f})"
            )

        return "<br>".join(stats)

    def find_similar_terms(
        self, query_term: str, n_neighbors: int = 10, distance_threshold: float = None
    ) -> List[Tuple[str, str, float, float]]:
        """Find terms similar to a query term in t-SNE space.

        Args:
            query_term: Term to find neighbors for.
            n_neighbors: Number of neighbors to return.
            distance_threshold: Maximum distance threshold.

        Returns:
            List of (term, source, score, distance) tuples.
        """
        if self.tsne_results is None:
            raise ValueError("No t-SNE results available")

        # Find the query term
        query_idx = None
        for i, item in enumerate(self.data):
            if item["term"].lower() == query_term.lower():
                query_idx = i
                break

        if query_idx is None:
            raise ValueError(f"Term '{query_term}' not found in dataset")

        # Calculate distances to all other points (works for both 2D and 3D)
        query_point = self.tsne_results[query_idx]
        distances = np.linalg.norm(self.tsne_results - query_point, axis=1)

        # Get nearest neighbors (excluding the query term itself)
        neighbor_indices = np.argsort(distances)[1 : n_neighbors + 1]

        neighbors = []
        for idx in neighbor_indices:
            item = self.data[idx]
            distance = distances[idx]

            if distance_threshold is None or distance <= distance_threshold:
                neighbors.append(
                    (item["term"], item["source"], item["score"], distance)
                )

        return neighbors


def main() -> None:
    """Main entry point for embedding visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize term embeddings using 2D or 3D t-SNE. "
        "Creates an interactive plot to explore how terms from different "
        "languages/sources cluster in the embedding space. Supports both "
        "2D and 3D visualizations with full interactivity."
    )
    parser.add_argument(
        "embedding_files",
        nargs="+",
        type=Path,
        help="JSONL embedding files to visualize",
    )
    parser.add_argument(
        "--max-terms-per-file",
        type=int,
        default=500,
        help="Maximum terms to load per file (default: 500)",
    )
    parser.add_argument(
        "--source-labels", nargs="+", help="Custom labels for each source file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output HTML file path (default: auto-generated)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of dimensions for t-SNE visualization (default: 2)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Multilingual Term Embeddings Visualization",
        help="Plot title",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter (default: 30.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of t-SNE iterations (default: 1000)",
    )
    parser.add_argument(
        "--learning-rate", default="auto", help="t-SNE learning rate (default: auto)"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip normalization before t-SNE"
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=8,
        help="Size of scatter plot points (default: 8)",
    )
    parser.add_argument(
        "--width", type=int, default=1200, help="Plot width in pixels (default: 1200)"
    )
    parser.add_argument(
        "--height", type=int, default=800, help="Plot height in pixels (default: 800)"
    )
    parser.add_argument(
        "--find-similar", type=str, help="Find terms similar to this query term"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)",
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
    for file_path in args.embedding_files:
        if not file_path.exists():
            logging.error(f"Input file not found: {file_path}")
            return

    # Generate output filename if not specified
    if args.output_file is None:
        output_filename = (
            f"embedding_visualization_{len(args.embedding_files)}files.html"
        )
        args.output_file = Path(output_filename)

    try:
        # Initialize visualizer
        visualizer = EmbeddingVisualizer(random_state=args.random_seed)

        # Load embeddings from multiple files
        visualizer.load_multiple_files(
            args.embedding_files,
            max_terms_per_file=args.max_terms_per_file,
            source_labels=args.source_labels,
        )

        # Apply t-SNE
        visualizer.apply_tsne(
            n_components=args.dimensions,
            perplexity=args.perplexity,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            normalize=not args.no_normalize,
        )

        # Create interactive plot
        fig = visualizer.create_interactive_plot(
            output_path=args.output_file,
            title=args.title,
            width=args.width,
            height=args.height,
            point_size=args.point_size,
        )

        # Show the plot
        fig.show()

        # Find similar terms if requested
        if args.find_similar:
            try:
                neighbors = visualizer.find_similar_terms(
                    args.find_similar, n_neighbors=10
                )
                print(f"\nTerms similar to '{args.find_similar}':")
                print(
                    f"{'Rank':<4} {'Term':<30} {'Source':<15} {'Score':<8} {'Distance':<8}"
                )
                print("-" * 70)
                for i, (term, source, score, distance) in enumerate(neighbors, 1):
                    print(
                        f"{i:<4} {term:<30} {source:<15} {score:<8.3f} {distance:<8.3f}"
                    )
            except ValueError as e:
                logging.error(f"Error finding similar terms: {e}")

        # Print summary
        print(f"\n{'='*60}")
        print("EMBEDDING VISUALIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Input files: {len(args.embedding_files)}")
        print(f"Total terms visualized: {len(visualizer.data)}")
        print(f"Embedding dimension: {visualizer.embeddings.shape[1]}")
        print(f"t-SNE dimensions: {args.dimensions}D")
        print(f"t-SNE perplexity: {args.perplexity}")
        print(f"t-SNE max iterations: {args.max_iter}")
        print(f"Interactive {args.dimensions}D plot saved: {args.output_file}")

        # Show source breakdown
        sources = {}
        for item in visualizer.data:
            source = item["source"]
            sources[source] = sources.get(source, 0) + 1

        print(f"\nTerms by source:")
        for source, count in sources.items():
            print(f"  {source}: {count} terms")

    except KeyboardInterrupt:
        logging.info("Visualization interrupted by user")
    except Exception as e:
        logging.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
