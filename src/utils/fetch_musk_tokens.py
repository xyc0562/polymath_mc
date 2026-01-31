"""
Fetch token IDs for current Musk tweet 7-day markets.

Usage:
    python3 -m src.utils.fetch_musk_tokens
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.algo.musk_tweet_count.musk_tweet_count import (
    GammaAPIClient,
    XTrackerClient,
)


def main():
    """Fetch and display current 7-day Musk tweet market token IDs."""
    print("=" * 70)
    print("Fetching Musk Tweet 7-Day Market Token IDs")
    print("=" * 70)

    gamma = GammaAPIClient()
    xtracker = XTrackerClient()

    # Fetch markets (only 7-day events due to parse_counting_dates_from_title)
    markets = gamma.get_musk_tweet_markets(xtracker_client=xtracker)

    if not markets:
        print("\nNo active 7-day Musk tweet markets found!")
        return

    # Group by event
    events = {}
    for m in markets:
        event_title = m.get("eventTitle", "Unknown")
        if event_title not in events:
            events[event_title] = {
                "title": event_title,
                "start": m.get("countingStartDate"),
                "end": m.get("countingEndDate"),
                "xtracker_count": m.get("xtrackerCount"),
                "markets": []
            }
        events[event_title]["markets"].append(m)

    print(f"\nFound {len(events)} active 7-day event(s):\n")

    for event_title, event_data in events.items():
        start = event_data["start"]
        end = event_data["end"]
        count = event_data["xtracker_count"]

        print(f"Event: {event_title}")
        if start and end:
            print(f"  Counting: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Current XTracker count: {count}")
        print(f"  Markets: {len(event_data['markets'])}")
        print()

        # Sort markets by threshold (bin order)
        sorted_markets = sorted(
            event_data["markets"],
            key=lambda x: x.get("groupItemThreshold", 0)
        )

        # Collect token IDs
        token_ids = []
        print("  Bin            | Token ID (YES)")
        print("  " + "-" * 60)

        for m in sorted_markets:
            outcome = m.get("groupItemTitle", "Unknown")
            clob_ids = m.get("clobTokenIds", [])

            # Parse clobTokenIds if it's a string
            import json
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except:
                    clob_ids = []

            yes_token = clob_ids[0] if clob_ids else "N/A"
            token_ids.append(yes_token)

            # Truncate token ID for display
            token_display = yes_token[:40] + "..." if len(yes_token) > 40 else yes_token
            print(f"  {outcome:14} | {token_display}")

        # Print comma-separated list for .env
        print()
        print("  For .env file (comma-separated):")
        print("  MUSK_TWEET_TOKEN_IDS=" + ",".join(token_ids))
        print()
        print("=" * 70)


if __name__ == "__main__":
    main()
