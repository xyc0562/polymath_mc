"""
Polymarket pUSD collateral helpers.

Polymarket's CLOB V2 settles trades in pUSD (Polymarket USD), an ERC-20
wrapper backed 1:1 by USDC.e on Polygon. To trade, USDC.e must be wrapped
to pUSD via the CollateralOnramp contract:

  1. USDC_E.approve(COLLATERAL_ONRAMP, amount)
  2. COLLATERAL_ONRAMP.wrap(USDC_E_ADDRESS, recipient, amount)

This module provides the on-chain primitives. The corresponding operator
CLI is `scripts/wrap_usdce_to_pusd.py`. Wrapping is intentionally NOT
performed by the bot at startup — the bot fails fast if pUSD balance is
insufficient and points the operator at the CLI.

Addresses verified from docs.polymarket.com/concepts/pusd (2026-04-29).
"""

from __future__ import annotations

from web3 import Web3

USDC_E_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
COLLATERAL_ONRAMP_ADDRESS = Web3.to_checksum_address(
    "0x93070a847efEf7F70739046A929D47a521F5B8ee"
)
USDC_E_DECIMALS = 6
MAX_UINT256 = 2**256 - 1

ERC20_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "approve",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
]

ONRAMP_WRAP_ABI = [
    {
        "name": "wrap",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "_asset", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_amount", "type": "uint256"},
        ],
        "outputs": [],
    },
]


def _usdce(w3: Web3):
    return w3.eth.contract(address=USDC_E_ADDRESS, abi=ERC20_ABI)


def _onramp(w3: Web3):
    return w3.eth.contract(address=COLLATERAL_ONRAMP_ADDRESS, abi=ONRAMP_WRAP_ABI)


def get_usdce_balance(w3: Web3, wallet: str) -> int:
    return _usdce(w3).functions.balanceOf(Web3.to_checksum_address(wallet)).call()


def get_onramp_allowance(w3: Web3, wallet: str) -> int:
    return (
        _usdce(w3)
        .functions.allowance(Web3.to_checksum_address(wallet), COLLATERAL_ONRAMP_ADDRESS)
        .call()
    )


def _send(w3: Web3, tx, private_key: str) -> str:
    signed = w3.eth.account.sign_transaction(tx, private_key=private_key)
    raw = signed.raw_transaction if hasattr(signed, "raw_transaction") else signed.rawTransaction
    tx_hash = w3.eth.send_raw_transaction(raw)
    return tx_hash.hex()


def approve_onramp(
    w3: Web3,
    wallet: str,
    private_key: str,
    amount: int = MAX_UINT256,
) -> str:
    """USDC_E.approve(COLLATERAL_ONRAMP, amount). Returns tx hash hex."""
    wallet = Web3.to_checksum_address(wallet)
    nonce = w3.eth.get_transaction_count(wallet)
    fn = _usdce(w3).functions.approve(COLLATERAL_ONRAMP_ADDRESS, amount)
    tx = fn.build_transaction({
        "from": wallet,
        "nonce": nonce,
        "chainId": w3.eth.chain_id,
    })
    return _send(w3, tx, private_key)


def wrap_usdce_to_pusd(
    w3: Web3,
    wallet: str,
    private_key: str,
    amount_wei: int,
) -> str:
    """COLLATERAL_ONRAMP.wrap(USDC_E_ADDRESS, wallet, amount_wei). Returns tx hash hex."""
    wallet = Web3.to_checksum_address(wallet)
    nonce = w3.eth.get_transaction_count(wallet)
    fn = _onramp(w3).functions.wrap(USDC_E_ADDRESS, wallet, amount_wei)
    tx = fn.build_transaction({
        "from": wallet,
        "nonce": nonce,
        "chainId": w3.eth.chain_id,
    })
    return _send(w3, tx, private_key)


def to_usdce_wei(usd_amount: float) -> int:
    """Convert USD float (e.g. 100.0) to USDC.e raw uint256."""
    return int(round(usd_amount * (10**USDC_E_DECIMALS)))


def from_usdce_wei(wei: int) -> float:
    return wei / (10**USDC_E_DECIMALS)
