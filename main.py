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
        scrape_post_content()


def cmd_analyze(args):
    """Handle the analyze subcommand."""
    from analyzer import analyze_all_posts
    from db import init_db

    init_db()
    analyze_all_posts()


def cmd_rankings(args):
    """Handle the rankings subcommand."""
    from db import init_db, get_ranked_books

    init_db()
    get_ranked_books(top=args.top, genre=args.genre)


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
    scrape_parser.set_defaults(func=cmd_scrape)

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze scraped posts for book mentions and sentiment'
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
