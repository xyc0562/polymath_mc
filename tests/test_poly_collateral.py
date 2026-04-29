"""Unit tests for src.algo.musk_tweet_count.poly_collateral helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.algo.musk_tweet_count import poly_collateral as pc


def _fake_w3(usdce_balance: int = 0, allowance: int = 0):
    """Build a minimal w3 stub with usdce.balanceOf / usdce.allowance / wrap call recording."""
    w3 = MagicMock()
    w3.eth.chain_id = 137
    w3.eth.get_transaction_count.return_value = 7

    def contract(address, abi):
        c = MagicMock()
        c.functions.balanceOf.return_value.call.return_value = usdce_balance
        c.functions.allowance.return_value.call.return_value = allowance

        def build_tx(_args):
            return {"to": address, "from": _args.get("from"), "nonce": _args.get("nonce")}

        c.functions.approve.return_value.build_transaction.side_effect = lambda d: {**d, "data": "approve"}
        c.functions.wrap.return_value.build_transaction.side_effect = lambda d: {**d, "data": "wrap"}
        return c

    w3.eth.contract.side_effect = contract
    w3.eth.account.sign_transaction.return_value = SimpleNamespace(raw_transaction=b"\x01\x02")
    w3.eth.send_raw_transaction.return_value = bytes.fromhex("ab" * 32)
    return w3


def test_get_usdce_balance_returns_value_from_contract():
    w3 = _fake_w3(usdce_balance=123_456_789)
    assert pc.get_usdce_balance(w3, "0x1111111111111111111111111111111111111111") == 123_456_789


def test_get_onramp_allowance_returns_value_from_contract():
    w3 = _fake_w3(allowance=42)
    assert pc.get_onramp_allowance(w3, "0x1111111111111111111111111111111111111111") == 42


def test_to_and_from_usdce_wei_round_trip():
    assert pc.to_usdce_wei(100.0) == 100_000_000
    assert pc.from_usdce_wei(100_000_000) == 100.0
    assert pc.to_usdce_wei(0.000001) == 1


def test_approve_onramp_signs_and_sends():
    w3 = _fake_w3()
    tx_hash = pc.approve_onramp(w3, "0x" + "1" * 40, "0x" + "ab" * 32, amount=99)
    assert isinstance(tx_hash, str) and len(tx_hash) > 0
    w3.eth.send_raw_transaction.assert_called_once()


def test_wrap_signs_and_sends():
    w3 = _fake_w3(usdce_balance=10**9)
    tx_hash = pc.wrap_usdce_to_pusd(w3, "0x" + "1" * 40, "0x" + "ab" * 32, amount_wei=10**6)
    assert isinstance(tx_hash, str) and len(tx_hash) > 0
    w3.eth.send_raw_transaction.assert_called_once()


def test_addresses_are_polygon_mainnet():
    assert pc.USDC_E_ADDRESS == "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    assert pc.COLLATERAL_ONRAMP_ADDRESS == "0x93070a847efEf7F70739046A929D47a521F5B8ee"
