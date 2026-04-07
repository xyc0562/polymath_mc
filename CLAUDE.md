This document contains critical information about working with this codebase. Follow these guidelines precisely.

This is a backtest/live advisor integration for China A shares, written in Python.
./venv is python virtual environment

## Design choices:
  1. DO NOT overcomplicate things unless ABSOLUTELY necessary, and before doing so, ALWAYS ask for permission
  2. ALWAYS check for duplicated functionality and unnecessary functionality when you introduce new features. DO NOT introduce these
  3. Whenever we need to have configuration settings, store them in a single place. when we have cli options, default to read from the single place config. NEVER duplicate hardcoded configs

## Research methodoligies:
  1. Do not even consider hard arbitrage opportunities within Polymarket, it's impossible

## Operation related
- Run the tests in parallel if possible
