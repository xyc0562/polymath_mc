"""
Derive Polymarket API credentials from your wallet private key.

This script derives the API key, secret, and passphrase needed for trading.
Run this once to get your credentials, then add them to your .env file.

Usage:
    python3 -m src.utils.derive_poly_keys

The script will:
1. Load your private key (encrypted or plain) from environment
2. Connect to Polymarket and derive API credentials
3. Print the credentials for you to save

Environment variables (set in .env or shell):
    - POLYMARKET_PRIVATE_KEY: Plain private key (0x...)
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY: Encrypted key string
    - ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE: Path to encrypted key file
    - PK_PWD: Password for encrypted key (optional, will prompt if not set)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from py_clob_client_v2.client import ClobClient

from src.utils.crypto_utils import load_private_key


def derive_api_credentials(
    private_key: str,
    host: str = "https://clob.polymarket.com",
    chain_id: int = 137,
) -> dict:
    """
    Derive Polymarket API credentials from a private key.

    Args:
        private_key: Ethereum private key (with or without 0x prefix)
        host: Polymarket CLOB API host
        chain_id: Chain ID (137 for Polygon mainnet)

    Returns:
        Dict with apiKey, secret, passphrase
    """
    # Ensure private key has 0x prefix
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    # Create client with just private key (no creds yet)
    client = ClobClient(
        host=host,
        chain_id=chain_id,
        key=private_key,
    )

    # Derive or retrieve API credentials
    # This will create new credentials if none exist, or return existing ones
    creds = client.create_or_derive_api_key()

    return creds


def main():
    """Main entry point."""
    print("=" * 60)
    print("Polymarket API Credential Derivation")
    print("=" * 60)
    print()

    # Load .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded .env file")
    except ImportError:
        print("Note: python-dotenv not installed, using environment variables only")

    # Load private key
    print()
    print("Loading private key...")
    try:
        private_key = load_private_key()
        # Mask the key for display
        masked = private_key[:6] + "..." + private_key[-4:] if len(private_key) > 10 else "***"
        print(f"Private key loaded: {masked}")
    except ValueError as e:
        print(f"Error: {e}")
        print()
        print("Please set one of the following environment variables:")
        print("  - POLYMARKET_PRIVATE_KEY (plain key)")
        print("  - ENCRYPTED_POLYMARKET_PRIVATE_KEY (encrypted key)")
        print("  - ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE (path to encrypted key file)")
        sys.exit(1)

    # Derive credentials
    print()
    print("Deriving API credentials from Polymarket...")
    print("(This may take a few seconds)")
    print()

    try:
        creds = derive_api_credentials(private_key)
    except Exception as e:
        print(f"Error deriving credentials: {e}")
        sys.exit(1)

    # Display credentials
    # Handle both dict and ApiCreds object
    if hasattr(creds, "api_key"):
        # ApiCreds object
        api_key = creds.api_key
        api_secret = creds.api_secret
        passphrase = creds.api_passphrase
    else:
        # Dict response
        api_key = creds.get("apiKey") or creds.get("api_key")
        api_secret = creds.get("secret") or creds.get("api_secret")
        passphrase = creds.get("passphrase") or creds.get("api_passphrase")

    print("=" * 60)
    print("SUCCESS! Add these to your .env file:")
    print("=" * 60)
    print()
    print(f'POLY_API_KEY={api_key}')
    print(f'POLY_API_SECRET={api_secret}')
    print(f'POLY_PASSPHRASE={passphrase}')
    print()
    print("=" * 60)
    print()
    print("Note: These credentials are derived from your wallet.")
    print("They will remain the same as long as you use the same private key.")
    print()


if __name__ == "__main__":
    main()
