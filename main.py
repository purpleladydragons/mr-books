#!/usr/bin/env python3
"""
Marginal Revolution Book Reviews Scraper and Analyzer

Scrape Marginal Revolution book reviews, analyze sentiment,
and rank books by how positively Tyler Cowen feels about them.
"""

import argparse
import sys


def cmd_scrape(args):
    """Handle the scrape subcommand."""
    from scraper import scrape_category_listing, scrape_post_content
    from db import init_db

    init_db()

    if args.listing_only:
        scrape_category_listing()
    else:
        scrape_category_listing()
        scrape_post_content(workers=args.workers)


def cmd_analyze(args):
    """Handle the analyze subcommand."""
    from analyzer import analyze_all_posts
    from db import init_db

    init_db()
    analyze_all_posts(
        use_ollama=args.use_ollama,
        model=args.model,
        reextract=args.reextract,
        full_context=args.full_context
    )


def cmd_rankings(args):
    """Handle the rankings subcommand."""
    from db import init_db, get_ranked_books, get_ranked_books_by_bt, has_bt_scores

    init_db()

    # Use Bradley-Terry scores if available, otherwise fall back to sentiment scores
    if has_bt_scores():
        print("(Using Bradley-Terry pairwise rankings)\n")
        get_ranked_books_by_bt(top=args.top, genre=args.genre)
    else:
        print("(Using sentiment scores - run 'python main.py rank' for better rankings)\n")
        get_ranked_books(top=args.top, genre=args.genre)


def cmd_rank(args):
    """Handle the rank subcommand for Bradley-Terry pairwise ranking."""
    from analyzer import run_pairwise_ranking
    from db import init_db

    init_db()
    run_pairwise_ranking(
        n_comparisons=args.comparisons,
        model=args.model,
        workers=args.workers,
        adaptive=args.adaptive,
        top_k=args.top_k
    )


def cmd_status(args):
    """Handle the status subcommand."""
    from db import init_db, get_scrape_status

    init_db()
    get_scrape_status()


def main():
    parser = argparse.ArgumentParser(
        description='Marginal Revolution Book Reviews Scraper and Analyzer'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # scrape subcommand
    scrape_parser = subparsers.add_parser(
        'scrape',
        help='Scrape posts from Marginal Revolution'
    )
    scrape_parser.add_argument(
        '--listing-only',
        action='store_true',
        help='Only scrape category listing pages, not post content'
    )
    scrape_parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of concurrent workers for scraping post content (default: 5)'
    )
    scrape_parser.set_defaults(func=cmd_scrape)

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze scraped posts for book mentions and sentiment'
    )
    analyze_parser.add_argument(
        '--use-ollama',
        action='store_true',
        help='Use Ollama LLM for sentiment analysis instead of VADER'
    )
    analyze_parser.add_argument(
        '--model',
        type=str,
        default='llama3.2:3b',
        help='Ollama model to use (default: llama3.2:3b)'
    )
    analyze_parser.add_argument(
        '--reextract',
        action='store_true',
        help='Force re-extraction of books from all posts (ignore cache)'
    )
    analyze_parser.add_argument(
        '--full-context',
        action='store_true',
        help='Use full post content for LLM analysis (single LLM call per post with JSON output)'
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # rankings subcommand
    rankings_parser = subparsers.add_parser(
        'rankings',
        help='Display books ranked by sentiment score'
    )
    rankings_parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Limit results to top N books'
    )
    rankings_parser.add_argument(
        '--genre',
        type=str,
        default=None,
        help='Filter by genre'
    )
    rankings_parser.set_defaults(func=cmd_rankings)

    # status subcommand
    status_parser = subparsers.add_parser(
        'status',
        help='Check scraping progress'
    )
    status_parser.set_defaults(func=cmd_status)

    # rank subcommand for Bradley-Terry pairwise ranking
    rank_parser = subparsers.add_parser(
        'rank',
        help='Run Bradley-Terry pairwise ranking using LLM comparisons'
    )
    rank_parser.add_argument(
        '--comparisons',
        type=int,
        default=10000,
        help='Number of pairwise comparisons to make (default: 10000)'
    )
    rank_parser.add_argument(
        '--model',
        type=str,
        default='llama3.2:3b',
        help='Ollama model to use (default: llama3.2:3b)'
    )
    rank_parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Number of parallel workers for LLM calls (default: 5)'
    )
    rank_parser.add_argument(
        '--adaptive',
        action='store_true',
        help='Use adaptive/uncertainty sampling (focuses on uncertain rankings, refits model every 1000 comparisons)'
    )
    rank_parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Focus comparisons on identifying top K items for faster convergence (e.g., --top-k 100). Refits every 500 comparisons, stops early if top-k stabilizes.'
    )
    rank_parser.set_defaults(func=cmd_rank)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
