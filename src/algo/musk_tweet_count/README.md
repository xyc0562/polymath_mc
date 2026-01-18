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
- `cryptography` - For private key encryption/decryption

### Step 4: Set Up Your Private Key

The bot needs your wallet's private key to sign transactions. There are multiple methods, from simplest to most secure.

#### Method A: Plain Private Key (Simple, Less Secure)

Set the environment variable directly in your terminal:

```bash
export POLYMARKET_PRIVATE_KEY="0xYourPrivateKeyHere"
```

Or add to a `.env` file in the project root:

```bash
# .env file
POLYMARKET_PRIVATE_KEY=0xYourPrivateKeyHere
```

See [Security Warning](#security-warning) for why plain keys are risky.

#### Method B: Encrypted Private Key (Recommended)

This method encrypts your key with a password. You'll enter the password each time the bot starts.

**Step 1: Encrypt your private key**

```bash
# Interactive mode (prompts for key and password, both hidden)
python -m src.utils.crypto_utils encrypt
```

This outputs a base64-encoded encrypted string like:
```
Gk3J8f2mN...longBase64String...7xKp==
```

**Step 2: Save the encrypted key**

You have three options:

**Option 2a: Environment variable**
```bash
export ENCRYPTED_POLYMARKET_PRIVATE_KEY="Gk3J8f2mN...longBase64String...7xKp=="
```

**Option 2b: .env file (recommended)**

Add to your `.env` file:
```bash
# .env file
ENCRYPTED_POLYMARKET_PRIVATE_KEY=Gk3J8f2mN...longBase64String...7xKp==
```

**Option 2c: Separate file**

Save to a file and reference it:
```bash
# Save encrypted key to file
python -m src.utils.crypto_utils encrypt --output .encrypted_key

# Set environment variable pointing to file
export ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE=.encrypted_key

# Or in .env file:
# ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE=.encrypted_key
```

**Step 3: Run the bot**

When you run the bot, it will prompt for your password:

```bash
python -m src.algo.musk_tweet_count.musk_tweet_count --once
# Enter private key password: ********
```

**For testing/automation:** You can set `PK_PWD` to skip the prompt:

```bash
export PK_PWD="YourPassword"
# Or in .env: PK_PWD=YourPassword
```

(Only use this for testing with small amounts - it defeats the purpose of encryption!)

#### Example .env File

```bash
# .env file - add to .gitignore!

# Option 1: Plain key (less secure)
# POLYMARKET_PRIVATE_KEY=0xYourPrivateKeyHere

# Option 2: Encrypted key (recommended)
ENCRYPTED_POLYMARKET_PRIVATE_KEY=Gk3J8f2mN...base64...==

# Option 3: Encrypted key in separate file
# ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE=.encrypted_key

# Optional: Password for automation (less secure)
# PK_PWD=YourPassword
```

**Important:** Add `.env` and `.encrypted_key` to your `.gitignore` file!

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

### Why Plain Environment Variables Are Risky

Using `POLYMARKET_PRIVATE_KEY` directly is **convenient but not secure** because:

1. **Shell history**: Your key may be saved in `~/.bash_history` or `~/.zsh_history`
2. **Process list**: Other programs might see environment variables via `/proc`
3. **Memory dumps**: The key exists in plain text in RAM
4. **.env files**: If someone accesses your computer, they can read the file
5. **Shoulder surfing**: Someone might see your screen when you type the key

### Recommended: Use Encrypted Keys

The bot supports encrypted private keys via the `ENCRYPTED_POLYMARKET_PRIVATE_KEY` environment variable. This provides:

- **Password protection**: Even if someone gets the encrypted string, they can't use it without the password
- **Hidden input**: Password is entered with obfuscation (not visible when typing)
- **No history exposure**: The encryption tool uses `getpass`, which doesn't save to shell history

### Encryption/Decryption Tool

The project includes a utility for encrypting and decrypting private keys:

```bash
# Encrypt a private key (interactive, all inputs hidden)
python -m src.utils.crypto_utils encrypt

# Encrypt with arguments (for scripting/piping)
python -m src.utils.crypto_utils encrypt --key 0xYourKey --password YourPassword

# Pipe key from a file
cat private_key.txt | python -m src.utils.crypto_utils encrypt --key -

# Save encrypted key to file
python -m src.utils.crypto_utils encrypt --output encrypted_key.txt

# Decrypt (to verify your password works)
python -m src.utils.crypto_utils decrypt --encrypted "base64string..."

# Decrypt from file
python -m src.utils.crypto_utils decrypt --encrypted "$(cat encrypted_key.txt)"
```

### How the Bot Loads Keys

The bot automatically loads from `.env` file, then tries these methods in order:

1. **POLYMARKET_PRIVATE_KEY** - If set, uses this directly (less secure)
2. **ENCRYPTED_POLYMARKET_PRIVATE_KEY** - If set, decrypts it
3. **ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE** - If set, reads file and decrypts

For encrypted keys, password comes from:
- **PK_PWD** environment variable (for testing/automation)
- Interactive prompt (hidden input)

### Key Loading Priority

```
┌─────────────────────────────────────────────────────────────┐
│  Load .env file (if exists)                                 │
│                              ↓                              │
│  Is POLYMARKET_PRIVATE_KEY set?                             │
│  YES → Use it directly (less secure)                        │
│                              ↓ NO                           │
│  Is ENCRYPTED_POLYMARKET_PRIVATE_KEY set?                   │
│  YES → Decrypt with password ──────────────────────┐        │
│                              ↓ NO                  │        │
│  Is ENCRYPTED_POLYMARKET_PRIVATE_KEY_FILE set?     │        │
│  YES → Read file, decrypt with password ───────────┤        │
│                              ↓ NO                  │        │
│  Error: No private key found                       │        │
│                                                    ↓        │
│                              ┌──────────────────────────┐   │
│                              │ Get password:            │   │
│                              │ 1. Try PK_PWD env var    │   │
│                              │ 2. Prompt user (hidden)  │   │
│                              └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Additional Security Tips

1. **Use a separate wallet**: Create a dedicated wallet for trading with only the funds you're willing to risk.

2. **Limit funds**: Don't keep more money in the trading wallet than necessary.

3. **Use a hardware wallet**: For maximum security, use a hardware wallet and only transfer funds to the trading wallet when needed.

4. **Monitor your wallet**: Set up alerts for any transactions from your address.

5. **Run on a secure machine**: Use a dedicated computer or VM that you trust.

6. **Rotate keys**: If you suspect any compromise, transfer funds to a new wallet immediately.

---

## Troubleshooting

### "No private key found"

The bot couldn't find any private key. Set one of these:

```bash
# Option 1: Plain key (less secure)
export POLYMARKET_PRIVATE_KEY="0x..."

# Option 2: Encrypted key (recommended)
python -m src.utils.crypto_utils encrypt  # Follow prompts
export ENCRYPTED_POLYMARKET_PRIVATE_KEY="<output from above>"
```

### "Decryption failed: incorrect password or corrupted data"

Your password is wrong or the encrypted string is corrupted. Try:
1. Re-encrypt your key: `python -m src.utils.crypto_utils encrypt`
2. Verify the password works: `python -m src.utils.crypto_utils decrypt --encrypted "..."`

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
