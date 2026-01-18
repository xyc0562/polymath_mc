"""
Cryptographic utilities for secure private key handling.

Provides encryption and decryption of private keys using password-based
key derivation (PBKDF2) and Fernet symmetric encryption.
"""

import base64
import getpass
import os
import sys
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# Number of iterations for PBKDF2 (higher = more secure but slower)
PBKDF2_ITERATIONS = 480000

# Salt length in bytes
SALT_LENGTH = 16


def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """
    Derive a Fernet-compatible encryption key from a password.

    Args:
        password: The password string
        salt: Random salt bytes (16 bytes recommended)

    Returns:
        Base64-encoded 32-byte key suitable for Fernet
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_private_key(private_key: str, password: str) -> str:
    """
    Encrypt a private key using a password.

    The output format is: base64(salt + encrypted_data)

    Args:
        private_key: The private key to encrypt (e.g., "0x...")
        password: The password to encrypt with

    Returns:
        Base64-encoded string containing salt + encrypted data
    """
    # Generate random salt
    salt = os.urandom(SALT_LENGTH)

    # Derive encryption key from password
    key = derive_key_from_password(password, salt)

    # Encrypt the private key
    fernet = Fernet(key)
    encrypted = fernet.encrypt(private_key.encode())

    # Combine salt + encrypted data and encode as base64
    combined = salt + encrypted
    return base64.b64encode(combined).decode('ascii')


def decrypt_private_key(encrypted_data: str, password: str) -> str:
    """
    Decrypt a private key using a password.

    Args:
        encrypted_data: Base64-encoded string from encrypt_private_key()
        password: The password used during encryption

    Returns:
        The decrypted private key

    Raises:
        ValueError: If decryption fails (wrong password or corrupted data)
    """
    try:
        # Decode from base64
        combined = base64.b64decode(encrypted_data.encode('ascii'))

        # Extract salt and encrypted data
        salt = combined[:SALT_LENGTH]
        encrypted = combined[SALT_LENGTH:]

        # Derive encryption key from password
        key = derive_key_from_password(password, salt)

        # Decrypt
        fernet = Fernet(key)
        decrypted = fernet.decrypt(encrypted)

        return decrypted.decode()

    except InvalidToken:
        raise ValueError("Decryption failed: incorrect password or corrupted data")
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")


def get_password_from_prompt(prompt: str = "Enter password: ") -> str:
    """
    Securely prompt for a password (input is hidden).

    Args:
        prompt: The prompt message to display

    Returns:
        The entered password
    """
    return getpass.getpass(prompt)


def load_private_key(
    env_var: str = "POLYMARKET_PRIVATE_KEY",
    encrypted_env_var: str = "ENCRYPTED_POLYMARKET_PRIVATE_KEY",
    encrypted_file_env_var: str = "ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE",
    password_env_var: str = "PK_PWD",
    dotenv_path: Optional[str] = None,
) -> str:
    """
    Load private key from environment or .env file, with support for encrypted keys.

    Priority:
    1. If POLYMARKET_PRIVATE_KEY is set, use it directly
    2. If ENCRYPTED_POLYMARKET_PRIVATE_KEY is set, decrypt it
    3. If ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE is set, read and decrypt from file

    For encrypted keys, password is obtained from:
    - PK_PWD environment variable (for testing/automation)
    - Interactive prompt (hidden input)

    Supports .env files:
    - Automatically loads from .env in current directory
    - Or specify a custom path via dotenv_path parameter

    Args:
        env_var: Environment variable for plain private key
        encrypted_env_var: Environment variable for encrypted private key
        encrypted_file_env_var: Environment variable pointing to encrypted key file
        password_env_var: Environment variable for password (for testing)
        dotenv_path: Optional path to .env file (default: auto-detect)

    Returns:
        The private key string

    Raises:
        ValueError: If no private key is available or decryption fails
    """
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path)
    except ImportError:
        pass  # python-dotenv not installed, skip

    # Option 1: Plain private key from environment
    plain_key = os.environ.get(env_var)
    if plain_key:
        return plain_key

    # Option 2: Encrypted private key from environment variable
    encrypted_key = os.environ.get(encrypted_env_var)

    # Option 3: Encrypted private key from file
    if not encrypted_key:
        encrypted_file = os.environ.get(encrypted_file_env_var)
        if encrypted_file:
            try:
                with open(encrypted_file, 'r') as f:
                    encrypted_key = f.read().strip()
            except FileNotFoundError:
                raise ValueError(f"Encrypted key file not found: {encrypted_file}")
            except IOError as e:
                raise ValueError(f"Error reading encrypted key file: {e}")

    if encrypted_key:
        # Try password from environment first (for testing/automation)
        password = os.environ.get(password_env_var)

        if not password:
            # Prompt for password with hidden input
            password = get_password_from_prompt("Enter private key password: ")

        if not password:
            raise ValueError("Password is required to decrypt private key")

        return decrypt_private_key(encrypted_key, password)

    # No key available
    raise ValueError(
        f"No private key found. Set one of:\n"
        f"  - {env_var} (plain key, less secure)\n"
        f"  - {encrypted_env_var} (encrypted key string)\n"
        f"  - {encrypted_file_env_var} (path to encrypted key file)\n"
        f"\n"
        f"You can set these in a .env file in your project root."
    )


# =============================================================================
# CLI for encrypting/decrypting keys
# =============================================================================

def cli_encrypt():
    """CLI command to encrypt a private key."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Encrypt a private key with a password",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for key and password)
  python -m src.utils.crypto_utils encrypt

  # Provide key as argument
  python -m src.utils.crypto_utils encrypt --key 0xYourPrivateKey

  # Pipe key from file (password still prompted)
  cat private_key.txt | python -m src.utils.crypto_utils encrypt --key -

  # Provide both key and password (for scripting)
  python -m src.utils.crypto_utils encrypt --key 0xYourKey --password MyPassword
        """
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        help="Private key to encrypt. Use '-' to read from stdin."
    )
    parser.add_argument(
        "--password", "-p",
        type=str,
        help="Password for encryption. If not provided, will prompt securely."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file. If not provided, prints to stdout."
    )

    args = parser.parse_args()

    # Get private key
    if args.key == "-":
        # Read from stdin
        private_key = sys.stdin.read().strip()
    elif args.key:
        private_key = args.key.strip()
    else:
        # Prompt for key (hidden)
        private_key = getpass.getpass("Enter private key to encrypt: ")

    if not private_key:
        print("Error: No private key provided", file=sys.stderr)
        sys.exit(1)

    # Get password
    if args.password:
        password = args.password
    else:
        password = getpass.getpass("Enter encryption password: ")
        password_confirm = getpass.getpass("Confirm password: ")
        if password != password_confirm:
            print("Error: Passwords do not match", file=sys.stderr)
            sys.exit(1)

    if not password:
        print("Error: Password is required", file=sys.stderr)
        sys.exit(1)

    # Encrypt
    encrypted = encrypt_private_key(private_key, password)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(encrypted)
        print(f"Encrypted key written to: {args.output}", file=sys.stderr)
    else:
        print(encrypted)


def cli_decrypt():
    """CLI command to decrypt a private key."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Decrypt an encrypted private key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for encrypted key and password)
  python -m src.utils.crypto_utils decrypt

  # Provide encrypted key as argument
  python -m src.utils.crypto_utils decrypt --encrypted "base64string..."

  # Read encrypted key from file
  python -m src.utils.crypto_utils decrypt --encrypted "$(cat encrypted_key.txt)"

  # Pipe encrypted key from file
  cat encrypted_key.txt | python -m src.utils.crypto_utils decrypt --encrypted -
        """
    )
    parser.add_argument(
        "--encrypted", "-e",
        type=str,
        help="Encrypted key to decrypt. Use '-' to read from stdin."
    )
    parser.add_argument(
        "--password", "-p",
        type=str,
        help="Password for decryption. If not provided, will prompt securely."
    )

    args = parser.parse_args()

    # Get encrypted key
    if args.encrypted == "-":
        encrypted = sys.stdin.read().strip()
    elif args.encrypted:
        encrypted = args.encrypted.strip()
    else:
        encrypted = input("Enter encrypted key: ").strip()

    if not encrypted:
        print("Error: No encrypted key provided", file=sys.stderr)
        sys.exit(1)

    # Get password
    if args.password:
        password = args.password
    else:
        password = getpass.getpass("Enter decryption password: ")

    if not password:
        print("Error: Password is required", file=sys.stderr)
        sys.exit(1)

    # Decrypt
    try:
        decrypted = decrypt_private_key(encrypted, password)
        print(decrypted)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Private key encryption/decryption utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.utils.crypto_utils encrypt
  python -m src.utils.crypto_utils encrypt --key 0xYourKey
  python -m src.utils.crypto_utils decrypt --encrypted "base64..."
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Encrypt subcommand
    encrypt_parser = subparsers.add_parser(
        "encrypt",
        help="Encrypt a private key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    encrypt_parser.add_argument(
        "--key", "-k",
        type=str,
        help="Private key to encrypt. Use '-' to read from stdin."
    )
    encrypt_parser.add_argument(
        "--password", "-p",
        type=str,
        help="Password for encryption. If not provided, will prompt securely."
    )
    encrypt_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file. If not provided, prints to stdout."
    )

    # Decrypt subcommand
    decrypt_parser = subparsers.add_parser(
        "decrypt",
        help="Decrypt an encrypted key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    decrypt_parser.add_argument(
        "--encrypted", "-e",
        type=str,
        help="Encrypted key to decrypt. Use '-' to read from stdin."
    )
    decrypt_parser.add_argument(
        "--password", "-p",
        type=str,
        help="Password for decryption. If not provided, will prompt securely."
    )

    args = parser.parse_args()

    if args.command == "encrypt":
        # Get private key
        if args.key == "-":
            private_key = sys.stdin.read().strip()
        elif args.key:
            private_key = args.key.strip()
        else:
            private_key = getpass.getpass("Enter private key to encrypt: ")

        if not private_key:
            print("Error: No private key provided", file=sys.stderr)
            sys.exit(1)

        # Get password
        if args.password:
            password = args.password
        else:
            password = getpass.getpass("Enter encryption password: ")
            password_confirm = getpass.getpass("Confirm password: ")
            if password != password_confirm:
                print("Error: Passwords do not match", file=sys.stderr)
                sys.exit(1)

        if not password:
            print("Error: Password is required", file=sys.stderr)
            sys.exit(1)

        # Encrypt
        encrypted = encrypt_private_key(private_key, password)

        # Output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(encrypted)
            print(f"Encrypted key written to: {args.output}", file=sys.stderr)
        else:
            print(encrypted)

    elif args.command == "decrypt":
        # Get encrypted key
        if args.encrypted == "-":
            encrypted = sys.stdin.read().strip()
        elif args.encrypted:
            encrypted = args.encrypted.strip()
        else:
            encrypted = input("Enter encrypted key: ").strip()

        if not encrypted:
            print("Error: No encrypted key provided", file=sys.stderr)
            sys.exit(1)

        # Get password
        if args.password:
            password = args.password
        else:
            password = getpass.getpass("Enter decryption password: ")

        if not password:
            print("Error: Password is required", file=sys.stderr)
            sys.exit(1)

        # Decrypt
        try:
            decrypted = decrypt_private_key(encrypted, password)
            print(decrypted)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
