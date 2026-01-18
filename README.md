# Polymath

A collection of algorithmic trading bots for [Polymarket](https://polymarket.com), a decentralized prediction market platform.

## What is Polymarket?

Polymarket is a prediction market where users can trade on the outcomes of real-world events. Instead of betting on sports or casino games, you bet on questions like:

- "Will Bitcoin reach $100,000 by end of 2026?"
- "How many tweets will Elon Musk post this week?"
- "Will it rain in NYC tomorrow?"

Each outcome trades between $0.00 and $1.00. If your prediction is correct, your shares pay out $1.00. If wrong, they pay $0.00.

## What is Polymath?

Polymath is a framework for building automated trading strategies on Polymarket. Each "algo" (algorithm) in this project targets a specific type of market and implements its own logic for:

1. **Data Collection** - Fetching market prices, external data sources
2. **Fair Value Calculation** - Estimating the "true" probability of outcomes
3. **Trade Execution** - Placing orders when market prices diverge from fair value

## Project Structure

```
polymath/
├── README.md                 # This file - project overview
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files for each algo
│   └── musk_tweet_count.yaml
│
└── src/                      # Source code
    ├── __init__.py
    ├── const.py              # Global constants
    ├── exceptions.py         # Custom exception classes
    │
    ├── utils/                # Shared utilities
    │   ├── __init__.py
    │   ├── app_utils.py      # Logging, helpers
    │   └── crypto_utils.py   # Private key encryption/decryption
    │
    └── algo/                 # Trading algorithms
        ├── __init__.py
        │
        └── musk_tweet_count/ # Elon Musk tweet count markets
            ├── __init__.py
            ├── musk_tweet_count.py
            └── README.md     # Detailed documentation
```

## Available Algorithms

| Algorithm | Description | Documentation |
|-----------|-------------|---------------|
| **musk_tweet_count** | Trades on "How many tweets will Elon Musk post?" markets. Uses current tweet count from xtracker and time remaining to calculate fair values. | [README](src/algo/musk_tweet_count/README.md) |

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd polymath
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Your Private Key

Create a `.env` file in the project root:

**Option A: Plain key (simple, less secure)**
```bash
# .env file
POLYMARKET_PRIVATE_KEY=0xYourPrivateKeyHere
```

**Option B: Encrypted key (recommended)**
```bash
# Step 1: Encrypt your key (interactive, inputs are hidden)
python -m src.utils.crypto_utils encrypt

# Step 2: Add to .env file
# .env file
ENCRYPTED_POLYMARKET_PRIVATE_KEY=<output from step 1>

# When you run the bot, it will prompt for your password
```

**Option C: Encrypted key in separate file**
```bash
# Step 1: Encrypt and save to file
python -m src.utils.crypto_utils encrypt --output .encrypted_key

# Step 2: Reference in .env
# .env file
ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE=.encrypted_key
```

**Important:** Add `.env` and `.encrypted_key` to your `.gitignore`!

See the [Security Warning](#security-warning) section below.

### 4. Run an Algorithm

```bash
# Dry run (no real trades) - recommended first step
python -m src.algo.musk_tweet_count.musk_tweet_count --once --dry-run

# Single run with real trades
python -m src.algo.musk_tweet_count.musk_tweet_count --once

# Continuous trading
python -m src.algo.musk_tweet_count.musk_tweet_count
```

## Adding New Algorithms

To create a new trading algorithm:

1. Create a new folder under `src/algo/`:
   ```
   src/algo/your_algo_name/
   ├── __init__.py
   ├── your_algo_name.py
   └── README.md
   ```

2. Create a config file (optional):
   ```
   config/your_algo_name.yaml
   ```

3. Implement the core components:
   - **Data fetching** - Connect to relevant APIs
   - **Fair value calculation** - Your edge comes from better predictions
   - **Trade execution** - Use `py_clob_client` to interact with Polymarket

4. Document your algorithm in its README.md

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `py_clob_client` | Official Polymarket trading client |
| `requests` | HTTP requests for external APIs |
| `web3` / `eth-account` | Ethereum wallet operations |
| `python-dotenv` | Load environment variables from .env files |
| `ruamel.yaml` | Parse YAML configuration files |
| `cryptography` | Private key encryption/decryption |

## Security Warning

**Your private key controls your funds.** Anyone with access to it can drain your wallet.

### Risks of Plain Environment Variables

Using `POLYMARKET_PRIVATE_KEY` directly exposes your key to:
- Shell history (`~/.bash_history`, `~/.zsh_history`)
- Process inspection (`/proc` on Linux)
- Memory dumps
- Shoulder surfing

### Recommended: Use Encrypted Keys

The project includes an encryption utility that protects your key with a password:

```bash
# Encrypt your private key
python -m src.utils.crypto_utils encrypt
# Enter private key to encrypt: ******** (hidden)
# Enter encryption password: ******** (hidden)
# Confirm password: ******** (hidden)
# Output: Gk3J8f2mN...base64String...==

# Set the encrypted key (safe to have in shell history)
export ENCRYPTED_POLYMARKET_PRIVATE_KEY="Gk3J8f2mN...base64String...=="

# Bot will prompt for password at startup
python -m src.algo.musk_tweet_count.musk_tweet_count --once
# Enter private key password: ******** (hidden)
```

### Best Practices

- **Use a dedicated wallet** with limited funds for trading
- **Never commit keys** to Git (add to `.gitignore`)
- **Monitor your wallet** for unexpected transactions
- **Rotate keys** if you suspect compromise

See the [musk_tweet_count README](src/algo/musk_tweet_count/README.md#security-warning) for detailed security documentation.

## API References

- **Polymarket CLOB API**: Trading, orderbooks, orders
  - Docs: https://docs.polymarket.com
  - Client: `py_clob_client`

- **Gamma API**: Market discovery, metadata
  - Base URL: `https://gamma-api.polymarket.com`

- **Data API**: Positions, trade history
  - Base URL: `https://data-api.polymarket.com`

- **XTracker API**: Tweet counting (for Musk markets)
  - Base URL: `https://xtracker.polymarket.com/api`

## Disclaimer

This software is for educational and research purposes. Trading on prediction markets involves financial risk. The authors are not responsible for any losses incurred. Only trade with funds you can afford to lose.

## License

[Add your license here]
