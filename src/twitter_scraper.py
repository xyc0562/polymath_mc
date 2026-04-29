#!/usr/bin/env python3
"""
Twitter/X Tweet Scraper using twikit

Scrapes tweets from a specified Twitter user using date-windowed search
to bypass pagination limits. Stores results in a CSV file with incremental saving.

Usage:
    python twitter_scraper.py --username <handle> --from <YYYY-MM-DD> --until <YYYY-MM-DD> [--output <file.csv>]

First run will prompt for login credentials. Cookies are saved for subsequent runs.
"""

import asyncio
import argparse
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from getpass import getpass

from src.utils import twikit_patches  # noqa: F401  (must precede `from twikit import …`)
from twikit import Client


# Rate limiting configuration
MIN_DELAY_BETWEEN_REQUESTS = 15  # minimum seconds between API requests
MAX_DELAY_BETWEEN_REQUESTS = 25  # maximum seconds between API requests
TWEETS_PER_BATCH = 20  # number of tweets to fetch per request
LONG_PAUSE_INTERVAL = 100  # take a longer pause every N tweets
LONG_PAUSE_DURATION = 90  # seconds for longer pause
WINDOW_PAUSE_DURATION = 15  # seconds pause between date windows

# Date window configuration
DAYS_PER_WINDOW = 3  # days per search window

COOKIES_FILE = "twitter_cookies.json"
CSV_HEADERS = ["tweet_id", "post_date", "content", "type"]


def get_cookies_path() -> Path:
    """Get the path to the cookies file in the config directory."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir / COOKIES_FILE


def get_progress_path(username: str) -> Path:
    """Get the path to the progress file for tracking completed date ranges."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir / f"{username}_progress.json"


def load_progress(username: str) -> dict:
    """Load progress data for the given username."""
    progress_path = get_progress_path(username)
    if progress_path.exists():
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")
    return {"completed_windows": []}


def save_progress(username: str, progress: dict):
    """Save progress data for the given username."""
    progress_path = get_progress_path(username)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def get_tweet_type(tweet) -> str:
    """Determine the type of tweet (retweet or original)."""
    if tweet.retweeted_tweet is not None:
        return "retweet"
    return "original"


def load_existing_tweets(csv_path: Path) -> set:
    """Load existing tweet IDs from CSV to avoid duplicates."""
    existing_ids = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_ids.add(row["tweet_id"])
        except Exception as e:
            print(f"Warning: Could not read existing CSV: {e}")
    return existing_ids


def append_tweets_to_csv(csv_path: Path, tweets_data: list, is_new_file: bool):
    """Append tweets to CSV file, creating headers if needed."""
    mode = "w" if is_new_file else "a"
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if is_new_file:
            writer.writeheader()
        writer.writerows(tweets_data)


def format_tweet_content(tweet) -> str:
    """Format tweet content, handling retweets appropriately."""
    if tweet.retweeted_tweet is not None:
        # For retweets, include the original tweet's text
        rt_tweet = tweet.retweeted_tweet
        return f"RT @{rt_tweet.user.screen_name}: {rt_tweet.text}"
    return tweet.text


def generate_date_windows(start_date: datetime, end_date: datetime, days_per_window: int) -> list:
    """
    Generate date windows from start_date going back to end_date.
    Returns list of (window_start, window_end) tuples, most recent first.

    Each window spans exactly days_per_window days (inclusive of both ends).
    Windows do not overlap.

    Example with days_per_window=3, start=Jan 8, end=Jan 1:
      - Window 1: Jan 6 to Jan 8 (3 days)
      - Window 2: Jan 3 to Jan 5 (3 days)
      - Window 3: Jan 1 to Jan 2 (2 days, partial)
    """
    windows = []
    current_end = start_date

    while current_end >= end_date:
        # Calculate window start (days_per_window days including both start and end)
        current_start = current_end - timedelta(days=days_per_window - 1)
        if current_start < end_date:
            current_start = end_date

        windows.append((current_start, current_end))

        # Move to next window - end at day before current_start (no overlap)
        current_end = current_start - timedelta(days=1)

    return windows


def window_to_key(window_start: datetime, window_end: datetime) -> str:
    """Convert a date window to a string key for progress tracking."""
    return f"{window_start.strftime('%Y-%m-%d')}_{window_end.strftime('%Y-%m-%d')}"


def _load_first_cookie(cookies_path: Path) -> dict:
    """Load the first cookie from a cookies file (supports both single dict and array formats)."""
    with open(cookies_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError("Empty cookie list")
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid cookies format: expected dict or list")


async def login_client(client: Client, cookies_path: Path) -> bool:
    """Handle client login, using saved cookies if available."""
    if cookies_path.exists():
        try:
            cookie = _load_first_cookie(cookies_path)
            client.set_cookies(cookie)
            print("Loaded saved cookies successfully.")
            return True
        except Exception as e:
            print(f"Could not load cookies: {e}")
            print("Proceeding with fresh login...")

    # Fresh login required
    print("\n=== Twitter Login Required ===")
    print("Note: This scraper requires a Twitter account to access the API.")
    print("Your credentials are only used for authentication and cookies are saved locally.\n")

    auth_info_1 = input("Enter username or email: ").strip()
    auth_info_2 = input("Enter email or phone (optional, press Enter to skip): ").strip() or None
    password = getpass("Enter password: ")
    totp_secret = input("Enter 2FA TOTP secret (optional, press Enter to skip): ").strip() or None

    try:
        await client.login(
            auth_info_1=auth_info_1,
            auth_info_2=auth_info_2,
            password=password,
            totp_secret=totp_secret
        )
        # Save cookies for future use
        client.save_cookies(str(cookies_path))
        print("Login successful! Cookies saved for future use.")
        return True
    except Exception as e:
        print(f"Login failed: {e}")
        return False


async def scrape_window(
    client: Client,
    username: str,
    window_start: datetime,
    window_end: datetime,
    existing_ids: set,
    output_path: Path,
    is_new_file: bool,
    total_scraped_ref: list
) -> tuple[int, bool]:
    """
    Scrape tweets for a specific date window using search.

    Returns (new_tweets_count, is_new_file).
    """
    # Build search query with date range
    # Twitter search uses inclusive dates, so we add 1 day to end
    since_str = window_start.strftime("%Y-%m-%d")
    until_str = (window_end + timedelta(days=1)).strftime("%Y-%m-%d")
    query = f"from:{username} since:{since_str} until:{until_str}"

    print(f"\n{'='*50}")
    print(f"Searching: {query}")
    print(f"{'='*50}")

    window_new = 0
    batch_tweets = []
    first_request = True
    tweets = None
    consecutive_empty = 0

    while True:
        # Rate limiting delay
        if not first_request:
            delay = random.uniform(MIN_DELAY_BETWEEN_REQUESTS, MAX_DELAY_BETWEEN_REQUESTS)
            print(f"Waiting {delay:.1f}s before next request...")
            await asyncio.sleep(delay)

        # Long pause periodically
        if total_scraped_ref[0] > 0 and total_scraped_ref[0] % LONG_PAUSE_INTERVAL == 0:
            print(f"\nTaking a longer pause ({LONG_PAUSE_DURATION}s) to avoid rate limits...")
            await asyncio.sleep(LONG_PAUSE_DURATION)

        try:
            if first_request:
                tweets = await client.search_tweet(
                    query=query,
                    product="Latest",
                    count=TWEETS_PER_BATCH
                )
                first_request = False
            else:
                tweets = await tweets.next()

            if not tweets:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    print("No more tweets in this window.")
                    break
                continue

            consecutive_empty = 0

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                print(f"\nRate limited! Waiting 90 seconds...")
                await asyncio.sleep(90)
                continue
            else:
                print(f"\nError fetching tweets: {e}")
                break

        for tweet in tweets:
            total_scraped_ref[0] += 1

            # Skip if already in CSV
            if tweet.id in existing_ids:
                continue

            tweet_date = tweet.created_at_datetime

            # Extract tweet data
            tweet_data = {
                "tweet_id": tweet.id,
                "post_date": tweet_date.strftime("%Y-%m-%d %H:%M:%S"),
                "content": format_tweet_content(tweet),
                "type": get_tweet_type(tweet)
            }

            batch_tweets.append(tweet_data)
            existing_ids.add(tweet.id)
            window_new += 1

            # Print progress
            tweet_preview = tweet_data["content"][:50].replace("\n", " ")
            if len(tweet_data["content"]) > 50:
                tweet_preview += "..."
            print(f"[{total_scraped_ref[0]}] {tweet_data['post_date']} ({tweet_data['type']}): {tweet_preview}")

            # Save to CSV periodically
            if len(batch_tweets) >= 10:
                append_tweets_to_csv(output_path, batch_tweets, is_new_file)
                is_new_file = False
                print(f"  -> Saved {len(batch_tweets)} tweets to CSV")
                batch_tweets = []

    # Save remaining tweets
    if batch_tweets:
        append_tweets_to_csv(output_path, batch_tweets, is_new_file)
        is_new_file = False
        print(f"  -> Saved {len(batch_tweets)} tweets to CSV")

    return window_new, is_new_file


async def scrape_tweets_by_search(
    client: Client,
    username: str,
    from_date: datetime,
    until_date: datetime,
    output_path: Path
) -> int:
    """
    Scrape tweets from a user using date-windowed search.

    Returns the number of new tweets scraped.
    """
    print(f"\nStarting date-windowed search for @{username}")
    print(f"Date range: {from_date.strftime('%Y-%m-%d')} to {until_date.strftime('%Y-%m-%d')}")
    print(f"Window size: {DAYS_PER_WINDOW} days\n")

    # Load existing tweets and progress
    existing_ids = load_existing_tweets(output_path)
    is_new_file = len(existing_ids) == 0
    progress = load_progress(username)
    completed_windows = set(progress.get("completed_windows", []))

    if existing_ids:
        print(f"Found {len(existing_ids)} existing tweets in CSV, will skip duplicates.")
    if completed_windows:
        print(f"Found {len(completed_windows)} completed date windows, will skip them.")

    # Generate date windows (until_date is more recent, from_date is older)
    windows = generate_date_windows(until_date, from_date, DAYS_PER_WINDOW)

    print(f"\nTotal windows to process: {len(windows)}")
    windows_to_process = []
    for w_start, w_end in windows:
        key = window_to_key(w_start, w_end)
        if key not in completed_windows:
            windows_to_process.append((w_start, w_end, key))

    print(f"Windows remaining (not yet scraped): {len(windows_to_process)}")

    if not windows_to_process:
        print("\nAll date windows have been scraped already!")
        return 0

    total_new = 0
    total_scraped_ref = [0]  # Use list to pass by reference

    for i, (w_start, w_end, key) in enumerate(windows_to_process):
        print(f"\n>>> Processing window {i+1}/{len(windows_to_process)}: {w_start.strftime('%Y-%m-%d')} to {w_end.strftime('%Y-%m-%d')}")

        try:
            window_new, is_new_file = await scrape_window(
                client=client,
                username=username,
                window_start=w_start,
                window_end=w_end,
                existing_ids=existing_ids,
                output_path=output_path,
                is_new_file=is_new_file,
                total_scraped_ref=total_scraped_ref
            )

            total_new += window_new
            print(f"\nWindow complete: {window_new} new tweets")

            # Mark window as completed
            completed_windows.add(key)
            progress["completed_windows"] = list(completed_windows)
            save_progress(username, progress)
            print(f"Progress saved. Total new tweets so far: {total_new}")

            # Pause between windows
            if i < len(windows_to_process) - 1:
                print(f"\nPausing {WINDOW_PAUSE_DURATION}s before next window...")
                await asyncio.sleep(WINDOW_PAUSE_DURATION)

        except KeyboardInterrupt:
            print("\n\nInterrupted! Saving progress...")
            progress["completed_windows"] = list(completed_windows)
            save_progress(username, progress)
            raise

    return total_new


async def main():
    parser = argparse.ArgumentParser(
        description="Scrape tweets from a Twitter user using date-windowed search."
    )
    parser.add_argument(
        "--username", "-u",
        required=True,
        help="Twitter username/handle to scrape (without @)"
    )
    parser.add_argument(
        "--from", "-f",
        dest="from_date",
        required=True,
        help="Start date - oldest date to begin scraping from (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--until", "-t",
        required=True,
        help="End date - most recent date to scrape until (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV file path (default: <username>_tweets.csv)"
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Reset progress and re-scrape all date windows"
    )

    args = parser.parse_args()

    # Parse the dates
    try:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.from_date}'. Use YYYY-MM-DD format.")
        sys.exit(1)

    try:
        until_date = datetime.strptime(args.until, "%Y-%m-%d")
    except ValueError:
        print(f"Error: Invalid date format '{args.until}'. Use YYYY-MM-DD format.")
        sys.exit(1)

    # Validate date range
    if from_date >= until_date:
        print(f"Error: --from date ({args.from_date}) must be before --until date ({args.until}).")
        sys.exit(1)

    # Clean username (remove @ if present)
    username = args.username.lstrip("@")

    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"data/{username}_tweets.csv")

    # Reset progress if requested
    if args.reset_progress:
        progress_path = get_progress_path(username)
        if progress_path.exists():
            progress_path.unlink()
            print(f"Progress reset for @{username}")

    print("=" * 60)
    print("Twitter/X Tweet Scraper (Date-Windowed Search)")
    print("=" * 60)
    print(f"Target user: @{username}")
    print(f"Date range: {from_date.strftime('%Y-%m-%d')} to {until_date.strftime('%Y-%m-%d')}")
    print(f"Output file: {output_path}")
    print(f"Window size: {DAYS_PER_WINDOW} days")
    print("=" * 60)

    # Initialize client
    client = Client(language="en-US")
    cookies_path = get_cookies_path()

    # Login
    if not await login_client(client, cookies_path):
        print("Failed to authenticate. Exiting.")
        sys.exit(1)

    # Scrape tweets using search
    try:
        new_tweets = await scrape_tweets_by_search(client, username, from_date, until_date, output_path)

        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(f"New tweets scraped: {new_tweets}")
        print(f"Output saved to: {output_path.absolute()}")

        # Show total in CSV
        total_in_csv = len(load_existing_tweets(output_path))
        print(f"Total tweets in CSV: {total_in_csv}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        print("Run again to resume from where you left off.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("Progress has been saved. Run again to resume.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
