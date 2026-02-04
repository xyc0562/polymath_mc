To set allowance:

⏺ The update_balance_allowance API is a Polymarket backend service - you don't set the amount yourself. Polymarket's backend handles the ERC20 approval transaction and typically sets it to the maximum (2^256 - 1, essentially
  unlimited) so you don't need to approve again.

  You don't control the number - Polymarket does. This is standard practice for DeFi apps to avoid repeated approval transactions.

  Just run:

  python3 -c "
  from src.algo.musk_tweet_count.forecaster.run_multi_event import create_clob_client
  from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

  client = create_clob_client()

  # Check current state
  params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
  before = client.get_balance_allowance(params)
  print('Before:')
  print('  Balance:', int(before.get('balance', 0)) / 1e6, 'USDC')
  print('  Allowance:', int(before.get('allowance', 0)) / 1e6, 'USDC')

  # Update allowance
  print()
  print('Updating allowance...')
  result = client.update_balance_allowance(params)
  print('Result:', result)

  # Check after
  after = client.get_balance_allowance(params)
  print()
  print('After:')
  print('  Balance:', int(after.get('balance', 0)) / 1e6, 'USDC')
  print('  Allowance:', int(after.get('allowance', 0)) / 1e6, 'USDC')
  "

  The allowance should become a very large number after this.