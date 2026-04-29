"""
Wrap USDC.e to pUSD via Polymarket's CollateralOnramp.

Polymarket CLOB V2 settles in pUSD; USDC.e must be wrapped to pUSD before
trading. This is a one-time-per-deposit operator action — the bot never
auto-wraps.

Usage:
    python -m scripts.wrap_usdce_to_pusd --check
    python -m scripts.wrap_usdce_to_pusd --amount 100
    python -m scripts.wrap_usdce_to_pusd --amount all

Environment:
    POLYMARKET_PRIVATE_KEY (or ENCRYPTED_POLYMARKET_PRIVATE_KEY[_FILE])
    POLYGON_RPC_URL  (default: https://polygon-rpc.com)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from web3 import Web3

from src.algo.musk_tweet_count.poly_collateral import (
    COLLATERAL_ONRAMP_ADDRESS,
    USDC_E_ADDRESS,
    MAX_UINT256,
    approve_onramp,
    from_usdce_wei,
    get_onramp_allowance,
    get_usdce_balance,
    to_usdce_wei,
    wrap_usdce_to_pusd,
)
from src.utils.crypto_utils import load_private_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wrap_usdce")


def _confirm(prompt: str) -> bool:
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans == "y" or ans == "yes"


def _wait_for_receipt(w3: Web3, tx_hash: str, timeout: int = 180) -> dict:
    logger.info(f"Waiting for receipt {tx_hash} (up to {timeout}s)…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            if receipt is not None:
                return receipt
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"No receipt within {timeout}s for {tx_hash}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrap USDC.e to pUSD")
    parser.add_argument(
        "--amount",
        type=str,
        default=None,
        help='USD amount to wrap, e.g. "100" or "all"',
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print balances and allowance, no transactions",
    )
    parser.add_argument(
        "--rpc-url",
        default=os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com"),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    if not args.check and args.amount is None:
        parser.error("--amount is required unless --check is passed")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    private_key = load_private_key()
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not w3.is_connected():
        logger.error(f"Cannot connect to RPC: {args.rpc_url}")
        return 1

    account = w3.eth.account.from_key(private_key)
    wallet = account.address
    logger.info(f"Wallet: {wallet}")
    logger.info(f"Chain ID: {w3.eth.chain_id}")
    logger.info(f"USDC.e:           {USDC_E_ADDRESS}")
    logger.info(f"CollateralOnramp: {COLLATERAL_ONRAMP_ADDRESS}")

    usdce_wei = get_usdce_balance(w3, wallet)
    allowance_wei = get_onramp_allowance(w3, wallet)
    logger.info(f"USDC.e balance: ${from_usdce_wei(usdce_wei):,.6f}")
    logger.info(f"Onramp allowance: {from_usdce_wei(allowance_wei):,.6f} USDC.e")

    if args.check:
        return 0

    if args.amount.lower() == "all":
        amount_wei = usdce_wei
        amount_usd = from_usdce_wei(amount_wei)
    else:
        amount_usd = float(args.amount)
        amount_wei = to_usdce_wei(amount_usd)

    if amount_wei <= 0:
        logger.error("Nothing to wrap (amount=0)")
        return 1
    if amount_wei > usdce_wei:
        logger.error(
            f"Insufficient USDC.e: want {from_usdce_wei(amount_wei):,.6f}, "
            f"have {from_usdce_wei(usdce_wei):,.6f}"
        )
        return 1

    if allowance_wei < amount_wei:
        logger.info(
            f"Allowance {from_usdce_wei(allowance_wei):,.6f} < {amount_usd:,.6f}; "
            f"approving max."
        )
        if not args.yes and not _confirm("Send approve(MAX) tx?"):
            logger.info("Aborted at approve step.")
            return 1
        tx_hash = approve_onramp(w3, wallet, private_key, amount=MAX_UINT256)
        logger.info(f"approve tx: {tx_hash}")
        receipt = _wait_for_receipt(w3, tx_hash)
        if receipt.get("status") != 1:
            logger.error(f"Approve failed: {receipt}")
            return 1
        logger.info("Approve confirmed.")

    if not args.yes and not _confirm(f"Send wrap({amount_usd:,.6f} USDC.e) tx?"):
        logger.info("Aborted at wrap step.")
        return 1
    tx_hash = wrap_usdce_to_pusd(w3, wallet, private_key, amount_wei)
    logger.info(f"wrap tx: {tx_hash}")
    receipt = _wait_for_receipt(w3, tx_hash)
    if receipt.get("status") != 1:
        logger.error(f"Wrap failed: {receipt}")
        return 1
    logger.info("Wrap confirmed.")

    final_usdce = get_usdce_balance(w3, wallet)
    logger.info(f"Final USDC.e: ${from_usdce_wei(final_usdce):,.6f}")
    logger.info("Done. Check pUSD balance via the bot or CLOB API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
