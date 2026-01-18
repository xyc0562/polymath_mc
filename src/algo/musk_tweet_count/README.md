# Musk Tweet Count Trading Bot

A trading bot for Polymarket's "Elon Musk Tweet Count" prediction markets. This bot automatically monitors markets, calculates fair values based on current tweet counts, and places trades when it identifies profitable opportunities.

## Table of Contents

1. [How It Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Running the Bot](#running-the-bot)
6. [Security Warning](#security-warning)

---

## How It Works

### The Big Picture

Imagine a betting market where people predict: "How many tweets will Elon Musk post between January 13-20?"

The market offers multiple ranges to bet on:
- 0-19 tweets
- 20-39 tweets
- 40-59 tweets
- ... and so on

Each range has a price between $0.00 and $1.00. If you buy "40-59" at $0.25 and Musk tweets 45 times, your contract pays out $1.00 (you profit $0.75). If he tweets 60 times, your contract pays $0.00 (you lose $0.25).

### What This Bot Does

The bot runs in a loop, performing these steps:

```
┌─────────────────────────────────────────────────────────────┐
│  1. FETCH DATA                                              │
│     - Get active Musk tweet markets from Polymarket         │
│     - Get current tweet count from xtracker.polymarket.com  │
│     - Get orderbook prices (what people are willing to      │
│       buy/sell at)                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. CALCULATE FAIR VALUES                                   │
│     - Based on current count + time remaining, estimate     │
│       probability of each outcome                           │
│     - Example: 100 tweets so far, 3 days left, averaging    │
│       50/day → expect ~250 total → "240-259" most likely    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. FIND OPPORTUNITIES                                      │
│     - Compare fair value vs market price                    │
│     - If fair value = 0.40 but market asks $0.30 → BUY     │
│       (10% "edge")                                          │
│     - If fair value = 0.30 but market bids $0.40 → SELL    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. EXECUTE TRADES                                          │
│     - Place limit orders when edge exceeds threshold        │
│     - Respect position limits (don't bet too much)          │
│     - Cancel stale orders from previous rounds              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Wait 5 min
                              │
                              └──────── Loop back to step 1
```

### Key Components

| Component | What It Does |
|-----------|--------------|
| **GammaAPIClient** | Fetches market data from Polymarket's Gamma API |
| **XTrackerClient** | Gets current tweet count from xtracker.polymarket.com |
| **FairValueCalculator** | Estimates probability of each outcome (placeholder - you can customize!) |
| **TradingEngine** | Decides when to buy/sell based on edge thresholds |
| **PolymarketTradingBot** | Orchestrates everything and executes trades |

### The Math: Fair Value Calculation

The default calculator uses a simple Gaussian (normal distribution) model:

1. **Calculate tweet rate**: `current_count / hours_elapsed`
2. **Project final count**: `current_count + (rate × hours_remaining)`
3. **Add uncertainty**: Standard deviation decreases as we get closer to the end
4. **Calculate probabilities**: For each range, compute P(lower ≤ final ≤ upper) using the normal CDF

**Example**:
- Current count: 150 tweets
- Time elapsed: 84 hours (3.5 days)
- Time remaining: 84 hours (3.5 days)
- Tweet rate: 150 / 84 = 1.79 tweets/hour
- Projected final: 150 + (1.79 × 84) = 300 tweets
- With uncertainty σ = 50, P(280-299) ≈ 15%, P(300-319) ≈ 15%, etc.

You can (and should!) improve this model with your own logic in `FairValueCalculator`.

---

## Prerequisites

- **Python 3.9+** installed on your computer
- **A Polymarket account** with funds deposited
- **Your wallet's private key** (the one connected to Polymarket)

---

## Setup

### Step 1: Clone or Download the Code

If you have the code in a folder called `polymath`, navigate to it:

```bash
cd /path/to/polymath
```

### Step 2: Create a Virtual Environment

A virtual environment is like a "sandbox" that keeps this project's Python packages separate from other projects. This prevents version conflicts.

**On macOS/Linux:**
```bash
# Create the virtual environment (only need to do this once)
python3 -m venv venv

# Activate it (do this every time you open a new terminal)
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
# Create the virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
# Create the virtual environment
python -m venv venv

# Activate it
venv\Scripts\Activate.ps1
```

When activated, you'll see `(venv)` at the start of your terminal prompt.

### Step 3: Install Dependencies

With your virtual environment activated:

```bash
pip install -r requirements.txt
```

This installs all required packages:
- `py_clob_client` - Polymarket's trading client
- `requests` - For making HTTP requests
- `python-dotenv` - For loading environment variables
- `ruamel.yaml` - For reading config files
- `web3`, `eth-account` - For blockchain/wallet operations

### Step 4: Set Up Your Private Key

The bot needs your wallet's private key to sign transactions.

**Option A: Using a `.env` file (simpler but less secure)**

Create a file named `.env` in the project root:

```bash
# In the polymath folder, create .env file
echo "POLYMARKET_PRIVATE_KEY=your_private_key_here" > .env
```

Replace `your_private_key_here` with your actual private key (starts with `0x`).

**Option B: Set it directly in your terminal (temporary, more secure)**

**macOS/Linux:**
```bash
export POLYMARKET_PRIVATE_KEY="0xYourPrivateKeyHere"
```

**Windows (Command Prompt):**
```cmd
set POLYMARKET_PRIVATE_KEY=0xYourPrivateKeyHere
```

**Windows (PowerShell):**
```powershell
$env:POLYMARKET_PRIVATE_KEY = "0xYourPrivateKeyHere"
```

Note: This only lasts for the current terminal session.

---

## Configuration

The bot's behavior is controlled by `config/musk_tweet_count.yaml`:

```yaml
trading:
  min_edge_to_buy: 0.05        # Only buy if fair value is 5%+ above market price
  min_edge_to_sell: 0.03       # Only sell if market price is 3%+ above fair value
  max_position_size_usd: 100   # Maximum $100 per single bet
  max_total_exposure_usd: 500  # Maximum $500 total across all bets
  check_interval_seconds: 300  # Check markets every 5 minutes
  dry_run: false               # Set to true to simulate without real trades

fair_value:
  historical_mean: 350         # Expected tweets per 7-day period
  historical_std: 100          # Standard deviation (uncertainty)
```

### Key Settings Explained

| Setting | What It Means |
|---------|---------------|
| `min_edge_to_buy` | How much "value" you need to see before buying. 0.05 = 5%. If fair value is $0.40 and market is $0.35, that's 14% edge → buy! |
| `max_position_size_usd` | Limits how much you bet on a single outcome. Protects against concentrating too much on one prediction. |
| `max_total_exposure_usd` | Total money at risk across all bets. If you have $500 limit and $400 in bets, you can only add $100 more. |
| `dry_run` | When `true`, the bot logs what it *would* do without actually trading. **Always start with this!** |

---

## Running the Bot

Make sure your virtual environment is activated (`source venv/bin/activate` or equivalent).

### Mode 1: Dry Run (Recommended First Step!)

See what the bot would do without risking any money:

```bash
python -m src.algo.musk_tweet_count.musk_tweet_count --once --dry-run
```

Or set `dry_run: true` in the config file:

```bash
python -m src.algo.musk_tweet_count.musk_tweet_count --once
```

### Mode 2: Single Run (Once)

Run one cycle and exit:

```bash
python -m src.algo.musk_tweet_count.musk_tweet_count --once
```

This will:
1. Fetch current markets and prices
2. Calculate fair values
3. Place any orders with sufficient edge
4. Print a summary and exit

### Mode 3: Continuous Run

Run indefinitely, checking every 5 minutes (or whatever `check_interval_seconds` is set to):

```bash
python -m src.algo.musk_tweet_count.musk_tweet_count
```

Press `Ctrl+C` to stop.

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--once` | Run single cycle and exit |
| `--dry-run` | Simulate trades without executing |
| `--config PATH` | Use a different config file |
| `--interval SECONDS` | Override check interval |
| `--max-position USD` | Override max position size |
| `--max-exposure USD` | Override max total exposure |
| `--min-edge DECIMAL` | Override min edge to buy (e.g., 0.05 for 5%) |
| `-v, --verbose` | Show detailed debug logs |

**Examples:**

```bash
# Dry run with verbose logging
python -m src.algo.musk_tweet_count.musk_tweet_count --once --dry-run -v

# Run continuously with custom limits
python -m src.algo.musk_tweet_count.musk_tweet_count --max-position 50 --max-exposure 200

# Single run with 10% minimum edge
python -m src.algo.musk_tweet_count.musk_tweet_count --once --min-edge 0.10
```

---

## Security Warning

### The Risk of Storing Private Keys

Your private key is like the master password to your wallet. Anyone with it can:
- Transfer all your funds
- Sign any transaction on your behalf
- Completely drain your account

**NEVER:**
- Share your private key with anyone
- Commit it to Git or any version control
- Store it in plain text on a shared computer
- Use it on a computer that might have malware

### Current Method: Environment Variables

The current setup (using `POLYMARKET_PRIVATE_KEY` environment variable) is **convenient but not secure** because:

1. **Shell history**: Your key may be saved in `~/.bash_history` or similar
2. **Process list**: Other programs might see environment variables
3. **Memory dumps**: The key exists in plain text in RAM
4. **.env files**: If someone accesses your computer, they can read the file

### Recommended: Encrypt Your Private Key

For better security, encrypt your private key and enter a password each time the bot starts.

**Step 1: Create an encrypted key file**

Create a Python script `encrypt_key.py`:

```python
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

def encrypt_key():
    # Get the private key (hidden input)
    private_key = getpass.getpass("Enter your private key: ")

    # Get a password to encrypt it
    password = getpass.getpass("Enter encryption password: ")
    password_confirm = getpass.getpass("Confirm password: ")

    if password != password_confirm:
        print("Passwords don't match!")
        return

    # Generate encryption key from password
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

    # Encrypt the private key
    f = Fernet(key)
    encrypted = f.encrypt(private_key.encode())

    # Save salt + encrypted key
    with open('.encrypted_key', 'wb') as file:
        file.write(salt + encrypted)

    print("Private key encrypted and saved to .encrypted_key")
    print("Add .encrypted_key to your .gitignore!")

if __name__ == "__main__":
    encrypt_key()
```

**Step 2: Modify the bot to decrypt on startup**

Add a decryption function that prompts for password:

```python
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

def load_encrypted_key(filepath='.encrypted_key'):
    """Load and decrypt the private key, prompting for password."""
    password = getpass.getpass("Enter decryption password: ")

    with open(filepath, 'rb') as file:
        data = file.read()

    salt = data[:16]
    encrypted = data[16:]

    # Recreate encryption key from password
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

    # Decrypt
    f = Fernet(key)
    return f.decrypt(encrypted).decode()
```

This way, even if someone accesses your computer, they cannot use your key without knowing the password.

### Additional Security Tips

1. **Use a separate wallet**: Create a dedicated wallet for trading with only the funds you're willing to risk.

2. **Limit funds**: Don't keep more money in the trading wallet than necessary.

3. **Use a hardware wallet**: For maximum security, use a hardware wallet and only transfer funds to the trading wallet when needed.

4. **Monitor your wallet**: Set up alerts for any transactions from your address.

5. **Run on a secure machine**: Use a dedicated computer or VM that you trust.

---

## Troubleshooting

### "POLYMARKET_PRIVATE_KEY environment variable not set"

Make sure you've set the environment variable in your current terminal session:
```bash
export POLYMARKET_PRIVATE_KEY="0x..."
```

### "No active Musk tweet markets found"

This happens when there are no active weekly tweet counting markets. Polymarket creates new ones each week.

### "No tracking found for period"

The xtracker service might not have data for the current period yet. This usually resolves itself within a few hours of a new counting period starting.

### "Failed to fetch orderbook"

Some markets may not have active orderbooks yet. The bot will skip these and continue with others.

---

## Customizing the Fair Value Calculator

The default `FairValueCalculator` is a simple placeholder. To improve it:

1. Analyze historical tweet patterns (time of day, day of week effects)
2. Factor in news events or Musk's recent activity
3. Use more sophisticated statistical models

Edit the `calculate_fair_prices` method in `musk_tweet_count.py` to implement your own logic.

---

## Disclaimer

This software is provided for educational purposes. Trading on prediction markets involves financial risk. Past performance does not guarantee future results. Only trade with money you can afford to lose.
