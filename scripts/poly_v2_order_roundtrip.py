"""
End-to-end live order round-trip on Polymarket CLOB V2.

Posts a tiny GTC+postOnly buy at a price well below the best bid (so it
will rest but never fill), verifies the response, fetches open orders to
confirm it's resting, then cancels via cancel_order(OrderPayload(...))
and verifies the cancellation. Exercises the live wire path that the
read-only smoke and dry-run cannot:

  * create_and_post_order (POST /order)
  * post_only=True branch                       ← gates v2 server-side
  * cancel_order(OrderPayload(orderID=...))     ← v2 signature change
  * get_open_orders shape (sample row keys)

Defaults to a $1 BUY at price 0.01 on an active Musk-tweet token.
Prompts before sending the order — pass --yes to skip.

Safety:
- post_only=True means the exchange rejects the order if it would cross
  the spread (so we cannot accidentally market-buy at any price).
- price=0.01 is the CLOB minimum tick; far below any realistic best bid.
- If anything errors after the order is on the book, the orderID is
  printed loudly so you can cancel via the script's --cancel-id arg
  (or the Polymarket UI / cancel_all from a Python REPL).

Usage:
    PK_PWD=… venv/bin/python -m scripts.poly_v2_order_roundtrip
    PK_PWD=… venv/bin/python -m scripts.poly_v2_order_roundtrip --yes
    PK_PWD=… venv/bin/python -m scripts.poly_v2_order_roundtrip --cancel-id 0xabc…
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("poly_v2_roundtrip")


def build_client():
    """Mirror the smoke harness — same env-var contract as the bots."""
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import ApiCreds
    from py_clob_client_v2.constants import POLYGON

    from src.utils.crypto_utils import load_private_key

    pk = load_private_key()
    if not pk.startswith("0x"):
        pk = "0x" + pk

    host = os.environ.get("CLOB_HOST", "https://clob.polymarket.com")
    chain_id = int(os.environ.get("CHAIN_ID", str(POLYGON)))
    funder = os.environ.get("POLY_FUNDER") or None
    sig_type_str = os.environ.get("POLY_SIGNATURE_TYPE")
    signature_type = int(sig_type_str) if sig_type_str else None

    api_key = os.environ.get("CLOB_API_KEY") or os.environ.get("POLY_API_KEY")
    api_secret = os.environ.get("CLOB_API_SECRET") or os.environ.get("POLY_API_SECRET")
    api_passphrase = os.environ.get("CLOB_API_PASSPHRASE") or os.environ.get("POLY_PASSPHRASE")
    creds = None
    if api_key and api_secret and api_passphrase:
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)

    kwargs = {"host": host, "chain_id": chain_id, "key": pk}
    if creds is not None:
        kwargs["creds"] = creds
    if signature_type is not None:
        kwargs["signature_type"] = signature_type
    if funder:
        kwargs["funder"] = funder

    client = ClobClient(**kwargs)
    if creds is None:
        creds = client.create_or_derive_api_key()
        client.set_api_creds(creds)
        logger.info("derived API credentials from wallet")
    return client


def _has_live_orderbook(client, token_id: str) -> bool:
    """The exchange returns 'orderbook does not exist' for tokens whose books
    weren't migrated to V2 (or which have already settled). Probe before posting.
    """
    try:
        book = client.get_order_book(token_id)
    except Exception:
        return False
    if book is None:
        return False
    # OrderBookSummary may be a dataclass or dict; treat any non-empty side as live.
    bids = getattr(book, "bids", None) or (book.get("bids") if isinstance(book, dict) else None)
    asks = getattr(book, "asks", None) or (book.get("asks") if isinstance(book, dict) else None)
    return bool(bids) or bool(asks)


def pick_token(client) -> tuple[str, float]:
    """
    Find one active musk-event token whose CLOB orderbook is live.

    Iterates active events tagged Musk-tweet (972), skips events that
    settle in the next 24h, and probes each token via get_order_book
    until we find one the exchange will accept POST /order against.
    """
    import requests
    from datetime import datetime, timezone

    resp = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"tag_id": 972, "active": "true", "closed": "false", "limit": 25},
        timeout=10,
    )
    resp.raise_for_status()
    events = resp.json()
    if not events:
        raise RuntimeError("no active musk events found")

    now = datetime.now(timezone.utc)
    candidates: list[tuple[str, str, str, str]] = []  # (event_title, mkt_q, token_id, end_date)

    for ev in events:
        title = ev.get("title", "")
        if "musk" not in title.lower():
            continue
        end = ev.get("endDate") or ""
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            if (end_dt - now).total_seconds() < 24 * 3600:
                continue  # too close to settlement
        except Exception:
            continue
        for m in ev.get("markets", []):
            ids = m.get("clobTokenIds")
            if isinstance(ids, str):
                try:
                    ids = json.loads(ids)
                except Exception:
                    continue
            if isinstance(ids, list) and ids:
                candidates.append((title, m.get("question", "?"), str(ids[0]), end))

    logger.info(f"discovered {len(candidates)} candidate tokens; probing for live books…")
    for title, q, tid, end in candidates:
        if not _has_live_orderbook(client, tid):
            continue
        tick = client.get_tick_size(tid)
        logger.info(f"event:  {title}  (ends {end})")
        logger.info(f"market: {q[:80]}")
        logger.info(f"token:  {tid}")
        logger.info(f"tick:   {tick}")
        return tid, float(tick)

    raise RuntimeError(
        "no candidate token has a live V2 orderbook — try with a specific "
        "--token-id argument from the bot's --list-events output"
    )


def post_test_order(client, token_id: str, tick_size: float) -> dict:
    from py_clob_client_v2.clob_types import (
        OrderArgs,
        OrderType,
        PartialCreateOrderOptions,
    )

    # Tick-aligned price well below any realistic bid. With post_only=True
    # the order rests if and only if it doesn't cross the spread.
    price = max(0.01, tick_size)
    # Notional = price * size. We want >$1 to clear the exchange min.
    # At price=0.01, size=200 → notional $2.00.
    size = max(15.0, 2.0 / price)
    notional = price * size

    args = OrderArgs(token_id=token_id, price=price, size=size, side="BUY")
    logger.info(f"order:  BUY {size:.0f} @ {price:.4f} = ${notional:.2f} (GTC, post_only=True)")

    return client.create_and_post_order(
        args,
        options=PartialCreateOrderOptions(tick_size=str(tick_size)),
        order_type=OrderType.GTC,
        post_only=True,
    )


def cancel_one(client, order_id: str) -> dict:
    from py_clob_client_v2.clob_types import OrderPayload
    return client.cancel_order(OrderPayload(orderID=order_id))


def confirm_or_exit(prompt: str, auto_yes: bool) -> None:
    if auto_yes:
        return
    if input(f"{prompt} [y/N]: ").strip().lower() not in ("y", "yes"):
        logger.info("aborted by operator.")
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 live order round-trip")
    parser.add_argument("--yes", action="store_true", help="skip the post confirmation prompt")
    parser.add_argument(
        "--cancel-id",
        type=str,
        default=None,
        help="Cancel a specific order ID instead of running the round-trip "
        "(for recovery if a previous run left an order resting).",
    )
    parser.add_argument(
        "--token-id",
        type=str,
        default=None,
        help="Skip discovery and use a specific token ID (e.g. one printed by "
        "the bot's --list-events output).",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    client = build_client()
    logger.info(f"L1 address: {client.get_address()}")

    if args.cancel_id:
        logger.info(f"cancel-only mode for orderID={args.cancel_id}")
        resp = cancel_one(client, args.cancel_id)
        logger.info(f"cancel response: {resp}")
        return 0

    if args.token_id:
        token_id = args.token_id
        tick = float(client.get_tick_size(token_id))
        if not _has_live_orderbook(client, token_id):
            logger.error(f"token {token_id} has no live V2 orderbook")
            return 6
        logger.info(f"token:  {token_id}  (tick={tick}, override)")
    else:
        token_id, tick = pick_token(client)
    confirm_or_exit("Post the test order?", args.yes)

    logger.info("===== POST =====")
    try:
        resp = post_test_order(client, token_id, tick)
    except Exception as e:
        logger.error(f"post failed: {type(e).__name__}: {e}")
        return 2
    logger.info(f"post response: {resp}")

    order_id = None
    if isinstance(resp, dict):
        order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
    if not order_id:
        logger.error(f"no orderID in response — full payload: {resp!r}")
        return 3
    logger.info(f"orderID: {order_id}")

    # Confirm it's resting on the book.
    logger.info("===== GET OPEN ORDERS =====")
    try:
        opens = client.get_open_orders()
    except Exception as e:
        logger.error(f"get_open_orders failed: {type(e).__name__}: {e}")
        logger.error(f"!! ORDER IS LIVE ON THE BOOK !! orderID={order_id}")
        logger.error(f"!! Recover with:  python -m scripts.poly_v2_order_roundtrip --cancel-id {order_id}")
        return 4
    matching = [o for o in opens if isinstance(o, dict) and o.get("id") == order_id]
    if matching:
        m = matching[0]
        logger.info(f"order on book: side={m.get('side')} price={m.get('price')} size={m.get('size_matched','?')}/{m.get('original_size','?')}")
    else:
        logger.warning(f"order {order_id} not found in {len(opens)} open orders (possible immediate match/kill)")

    # Cancel.
    logger.info("===== CANCEL =====")
    try:
        cresp = cancel_one(client, order_id)
    except Exception as e:
        logger.error(f"cancel failed: {type(e).__name__}: {e}")
        logger.error(f"!! Recover with:  python -m scripts.poly_v2_order_roundtrip --cancel-id {order_id}")
        return 5
    logger.info(f"cancel response: {cresp}")

    # Verify it's gone.
    time.sleep(2)
    try:
        opens_after = client.get_open_orders()
        still_there = any(
            isinstance(o, dict) and o.get("id") == order_id for o in opens_after
        )
        if still_there:
            logger.warning(f"order {order_id} still in open_orders after cancel; check Polymarket UI")
        else:
            logger.info(f"order {order_id} no longer in open_orders ✓")
    except Exception as e:
        logger.warning(f"post-cancel get_open_orders failed: {e}")

    logger.info("ROUND-TRIP OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
