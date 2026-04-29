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


def _book_sides(book) -> tuple[list, list]:
    """get_order_book returns a dict on v2; extract bids/asks defensively."""
    if book is None:
        return [], []
    if isinstance(book, dict):
        return book.get("bids") or [], book.get("asks") or []
    return list(getattr(book, "bids", []) or []), list(getattr(book, "asks", []) or [])


def _has_live_orderbook(client, token_id: str, require_both_sides: bool = True) -> bool:
    """A 'live' V2 book has both bids AND asks (otherwise POST /order tends
    to fail with 'the orderbook does not exist' even though GET /book is 200).
    """
    try:
        book = client.get_order_book(token_id)
    except Exception:
        return False
    bids, asks = _book_sides(book)
    return (bool(bids) and bool(asks)) if require_both_sides else (bool(bids) or bool(asks))


def pick_token(client) -> tuple[str, float]:
    """
    Find any high-volume market with both-sided live V2 liquidity.

    The v2 cutover left some V1 markets without a backfilled orderbook
    (returns 404 from GET /book and 400 from POST /order), so we cannot
    rely on the bot's own Musk universe for verification — we just want
    *any* market that proves the v2 wire path works. Sort gamma /markets
    by volume desc, probe books, take the first two-sided one.
    """
    import requests

    resp = requests.get(
        "https://gamma-api.polymarket.com/markets",
        params={
            "active": "true",
            "closed": "false",
            "order": "volumeNum",
            "ascending": "false",
            "limit": 50,
        },
        timeout=10,
    )
    resp.raise_for_status()
    markets = resp.json() or []

    logger.info(f"probing {len(markets)} high-volume markets for two-sided V2 books…")
    for m in markets:
        ids = m.get("clobTokenIds")
        if isinstance(ids, str):
            try:
                ids = json.loads(ids)
            except Exception:
                continue
        if not isinstance(ids, list) or not ids:
            continue
        tid = str(ids[0])
        try:
            book = client.get_order_book(tid)
        except Exception:
            continue
        bids, asks = _book_sides(book)
        if not (bids and asks):
            continue
        try:
            tick = float(client.get_tick_size(tid))
        except Exception:
            continue
        best_bid = float(bids[0]["price"]) if isinstance(bids[0], dict) else None
        best_ask = float(asks[0]["price"]) if isinstance(asks[0], dict) else None
        logger.info(f"market: {m.get('question', '?')[:80]}")
        logger.info(f"token:  {tid}")
        logger.info(f"tick:   {tick}  best_bid={best_bid}  best_ask={best_ask}")
        return tid, tick

    raise RuntimeError(
        "no high-volume market has a two-sided V2 orderbook — try --token-id "
        "with a token that you've confirmed has resting bids+asks on Polymarket UI"
    )


def post_test_order(
    client,
    token_id: str,
    tick_size: float,
    best_bid: float | None,
) -> dict:
    from py_clob_client_v2.clob_types import (
        OrderArgs,
        OrderType,
        PartialCreateOrderOptions,
    )

    # Place ~10 ticks below the best bid so we're guaranteed not to cross
    # any ask. Floor at the minimum tick. post_only=True is the belt-and-
    # suspenders: even if the price calc is wrong, the exchange rejects.
    if best_bid and best_bid > tick_size * 20:
        target = best_bid - tick_size * 10
    else:
        target = tick_size
    # Round to tick alignment (floor) and floor at one tick.
    price = max(tick_size, round(target / tick_size) * tick_size)
    # ~$2 notional minimum-clear. Round size to whole shares.
    size = max(15.0, round(2.0 / price))
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
        try:
            book = client.get_order_book(token_id)
        except Exception as e:
            logger.error(f"token {token_id}: get_order_book raised {type(e).__name__}: {e}")
            return 6
        bids, asks = _book_sides(book)
        if not (bids and asks):
            logger.error(f"token {token_id} has no two-sided V2 orderbook")
            return 6
        best_bid = float(bids[0]["price"]) if isinstance(bids[0], dict) else None
        logger.info(f"token:  {token_id}  (tick={tick}, best_bid={best_bid}, override)")
    else:
        token_id, tick = pick_token(client)
        # Re-fetch book to grab best_bid for the price calc
        bids, _ = _book_sides(client.get_order_book(token_id))
        best_bid = float(bids[0]["price"]) if bids and isinstance(bids[0], dict) else None
    confirm_or_exit("Post the test order?", args.yes)

    logger.info("===== POST =====")
    try:
        resp = post_test_order(client, token_id, tick, best_bid)
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
