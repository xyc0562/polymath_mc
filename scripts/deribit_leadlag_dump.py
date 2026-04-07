"""
Diagnostic dump for Deribit lead-lag pipeline.

Fetches live data from both Deribit and Polymarket, then dumps
detailed per-bin breakdowns to data/dumps/ for inspection.

Usage:
    python -m scripts.deribit_leadlag_dump
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algo.deribit_leadlag.deribit_client import fetch_btc_options_summary, fetch_spot_price
from src.algo.deribit_leadlag.polymarket_discovery import (
    discover_btc_threshold_markets,
    build_target_strikes,
)
from src.algo.deribit_leadlag.implied_probs import (
    build_call_price_curve,
    _find_bracketing_calls,
    digital_prob_from_call_spread,
    time_to_resolution_years,
)
from src.algo.deribit_leadlag.settlement import (
    CompatibilityClass,
    classify_compatibility,
)
from src.algo.deribit_leadlag.signal_comparator import (
    compute_polymarket_fee,
    interpolate_call_spread_to_poly_time,
)
from src.algo.deribit_leadlag.position_manager import BinKey

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEE_RATE = 0.072


def main():
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    dump_dir = Path("data/dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Polymarket markets
    logger.info("Fetching Polymarket BTC threshold markets...")
    poly_markets = discover_btc_threshold_markets()
    if not poly_markets:
        logger.error("No Polymarket markets found")
        return

    # Group by event (date)
    events = {}
    for m in poly_markets:
        events.setdefault(m.expiry_date, []).append(m)
    for d in events:
        events[d].sort(key=lambda m: m.strike)

    # 2. Fetch Deribit options
    logger.info("Fetching Deribit options...")
    options = fetch_btc_options_summary()
    if not options:
        logger.error("No Deribit options fetched")
        return

    spot = fetch_spot_price()

    # Classify settlement compatibility
    compat_map = {}
    for m in poly_markets:
        key = BinKey(expiry_date=m.expiry_date, strike=m.strike)
        if m.settlement is None:
            compat_map[key] = CompatibilityClass.REJECT
        else:
            compat_map[key], _ = classify_compatibility(m.expiry_date, m.settlement)

    eligible = sum(1 for v in compat_map.values() if v != CompatibilityClass.REJECT)
    logger.info(f"Settlement compatibility: {eligible}/{len(compat_map)} eligible")

    # Build call curves per expiry (include next-day for interpolation)
    target_strikes = build_target_strikes(poly_markets)
    expiry_dates = set(d for d, _ in target_strikes.keys())
    curves = {}
    for exp in expiry_dates:
        curves[exp] = build_call_price_curve(options, exp)
    # Also build next-day curves for time interpolation
    for exp in list(expiry_dates):
        next_day = exp + timedelta(days=1)
        if next_day not in curves:
            c = build_call_price_curve(options, next_day)
            if c:
                curves[next_day] = c

    # Get forwards per expiry
    forwards = {}
    for opt in options:
        if opt.option_type == "C" and opt.underlying_price > 0:
            forwards[opt.expiry_date] = opt.underlying_price

    # 3. Build detailed dump per event date
    all_output = []
    all_output.append(f"Deribit Lead-Lag Diagnostic Dump")
    all_output.append(f"Generated: {now.isoformat()}")
    all_output.append(f"Spot (Deribit index): ${spot:,.2f}" if spot else "Spot: N/A")
    all_output.append(f"Polymarket events: {len(events)} dates, {len(poly_markets)} total bins")
    all_output.append(f"Settlement eligible: {eligible}/{len(poly_markets)} bins")
    all_output.append(f"Deribit options fetched: {len(options)}")
    all_output.append("")

    json_dump = {
        "timestamp": now.isoformat(),
        "spot_price": spot,
        "events": {},
    }

    for exp_date in sorted(events.keys()):
        bins = events[exp_date]
        curve = curves.get(exp_date, [])
        forward = forwards.get(exp_date)
        resolution_utc = bins[0].resolution_time_utc

        T_years = time_to_resolution_years(now, resolution_utc)
        T_hours = T_years * 365.25 * 24

        all_output.append("=" * 100)
        all_output.append(f"EVENT: Bitcoin above ___ on {exp_date}")
        all_output.append(f"  Resolution: {resolution_utc.isoformat()}")
        all_output.append(f"  T to resolution: {T_hours:.1f} hours ({T_years:.5f} years)")
        all_output.append(f"  Forward (Deribit): ${forward:,.2f}" if forward else "  Forward: N/A")
        all_output.append(f"  Deribit call curve: {len(curve)} strikes")
        all_output.append(f"  Polymarket bins: {len(bins)}")
        all_output.append("")

        # Show the Deribit call curve for this expiry
        all_output.append(f"  --- Deribit Call Curve (expiry {exp_date}) ---")
        all_output.append(f"  {'Strike':>10}  {'Bid USD':>12}  {'Ask USD':>12}  {'Mid USD':>12}  {'Mark USD':>12}  {'Vol24h':>8}  {'OI':>8}  {'2-sided':>7}")
        for cp in curve:
            bid_s = f"${cp.bid_usd:>10,.2f}" if cp.bid_usd is not None else "        N/A"
            ask_s = f"${cp.ask_usd:>10,.2f}" if cp.ask_usd is not None else "        N/A"
            mid_s = f"${cp.mid_usd:>10,.2f}" if cp.mid_usd is not None else "        N/A"
            all_output.append(
                f"  {cp.strike:>10,.0f}  {bid_s}  {ask_s}  {mid_s}  ${cp.mark_usd:>10,.2f}  {cp.volume:>8.1f}  {cp.open_interest:>8.1f}  {'Y' if cp.has_two_sided_quotes else 'N':>7}"
            )
        all_output.append("")

        event_json = {
            "expiry_date": str(exp_date),
            "resolution_utc": resolution_utc.isoformat(),
            "T_hours": round(T_hours, 2),
            "forward": forward,
            "n_deribit_calls": len(curve),
            "bins": [],
        }

        # Per-bin detail
        all_output.append(f"  --- Per-Bin Analysis ---")

        for m in bins:
            strike = m.strike
            poly_yes = m.yes_price
            poly_no = m.no_price

            # Settlement compatibility
            bk = BinKey(expiry_date=m.expiry_date, strike=m.strike)
            compat = compat_map.get(bk, CompatibilityClass.REJECT)

            # Find bracketing calls
            bracket = _find_bracketing_calls(curve, strike)
            prob = digital_prob_from_call_spread(curve, strike, min_spread_usd=15.0)

            # Time-adjusted prob (the new model)
            curve_next = curves.get(exp_date + timedelta(days=1))
            adj_prob = None
            if compat != CompatibilityClass.REJECT:
                adj_prob = interpolate_call_spread_to_poly_time(
                    curve_same_day=curve,
                    curve_next_day=curve_next,
                    target_strike=strike,
                    min_spread_usd=15.0,
                )

            bin_json = {
                "strike": strike,
                "poly_yes_price": poly_yes,
                "poly_no_price": poly_no,
                "poly_volume": m.volume,
                "poly_condition_id": m.condition_id,
                "compatibility": compat.value,
            }

            all_output.append(f"  Strike ${strike:,.0f}  [{compat.value}]")
            all_output.append(f"    Polymarket: YES={poly_yes:.4f}  NO={poly_no:.4f}  vol=${m.volume:,.0f}")
            if m.settlement:
                s = m.settlement
                all_output.append(
                    f"    Settlement: {s.source_exchange or '?'} {s.source_symbol or '?'} "
                    f"{s.source_timeframe or '?'} {s.source_price_field or '?'} "
                    f"comparator={s.comparator or '?'} parse={'OK' if s.parse_success else 'FAIL'}"
                )

            if bracket is None:
                all_output.append(f"    Deribit: NO BRACKETING CALLS FOUND")
                bin_json["deribit_status"] = "no_bracketing_calls"
            else:
                lower, upper = bracket
                dk = upper.strike - lower.strike

                # Show the two reference options
                all_output.append(f"    Deribit bracketing calls:")
                all_output.append(
                    f"      Lower: K={lower.strike:>10,.0f}  bid=${lower.bid_usd or 0:>10,.2f}  ask=${lower.ask_usd or 0:>10,.2f}  "
                    f"mid=${lower.mid_usd or lower.mark_usd:>10,.2f}  mark=${lower.mark_usd:>10,.2f}  "
                    f"vol={lower.volume:.1f}  OI={lower.open_interest:.1f}  2-sided={'Y' if lower.has_two_sided_quotes else 'N'}"
                )
                all_output.append(
                    f"      Upper: K={upper.strike:>10,.0f}  bid=${upper.bid_usd or 0:>10,.2f}  ask=${upper.ask_usd or 0:>10,.2f}  "
                    f"mid=${upper.mid_usd or upper.mark_usd:>10,.2f}  mark=${upper.mark_usd:>10,.2f}  "
                    f"vol={upper.volume:.1f}  OI={upper.open_interest:.1f}  2-sided={'Y' if upper.has_two_sided_quotes else 'N'}"
                )
                all_output.append(f"      dK = ${dk:,.0f}")

                bin_json["deribit_lower"] = {
                    "strike": lower.strike,
                    "bid_usd": lower.bid_usd,
                    "ask_usd": lower.ask_usd,
                    "mid_usd": lower.mid_usd,
                    "mark_usd": lower.mark_usd,
                    "volume": lower.volume,
                    "open_interest": lower.open_interest,
                    "two_sided": lower.has_two_sided_quotes,
                }
                bin_json["deribit_upper"] = {
                    "strike": upper.strike,
                    "bid_usd": upper.bid_usd,
                    "ask_usd": upper.ask_usd,
                    "mid_usd": upper.mid_usd,
                    "mark_usd": upper.mark_usd,
                    "volume": upper.volume,
                    "open_interest": upper.open_interest,
                    "two_sided": upper.has_two_sided_quotes,
                }
                bin_json["dk"] = dk

            if prob is None:
                reason = "illiquid or spread too small/wide"
                if bracket is not None:
                    lower, upper = bracket
                    c_low = lower.mid_usd if lower.mid_usd is not None else lower.mark_usd
                    c_high = upper.mid_usd if upper.mid_usd is not None else upper.mark_usd
                    spread = c_low - c_high
                    all_output.append(f"    Implied prob: SKIPPED ({reason})")
                    all_output.append(f"      Raw spread: ${spread:,.2f} (min required: $15.00)")
                    if lower.has_two_sided_quotes and upper.has_two_sided_quotes:
                        dk = upper.strike - lower.strike
                        p_cons = max(0, min(1, (lower.bid_usd - upper.ask_usd) / dk))
                        p_agg = max(0, min(1, (lower.ask_usd - upper.bid_usd) / dk))
                        all_output.append(f"      Would-be bounds: [{p_cons:.3f}, {p_agg:.3f}] width={p_agg - p_cons:.3f}")
                else:
                    all_output.append(f"    Implied prob: SKIPPED ({reason})")
                bin_json["implied_prob"] = None
                bin_json["skip_reason"] = reason
            else:
                all_output.append(f"    Implied prob ({prob.method}):")
                all_output.append(f"      Conservative: {prob.prob_conservative:.4f}")
                all_output.append(f"      Mid:          {prob.prob_mid:.4f}")
                all_output.append(f"      Aggressive:   {prob.prob_aggressive:.4f}")
                all_output.append(f"      Bounds width: {prob.prob_aggressive - prob.prob_conservative:.4f}")

                bin_json["implied_prob"] = {
                    "conservative": round(prob.prob_conservative, 6),
                    "mid": round(prob.prob_mid, 6),
                    "aggressive": round(prob.prob_aggressive, 6),
                    "method": prob.method,
                    "bounds_width": round(prob.prob_aggressive - prob.prob_conservative, 6),
                }

                # Edge calculation (raw, same-day only)
                if prob.prob_mid > poly_yes:
                    side = "BUY_YES"
                    entry_price = poly_yes
                    conservative_prob = prob.prob_conservative
                    gross_edge = conservative_prob - poly_yes
                else:
                    side = "BUY_NO"
                    entry_price = poly_no
                    conservative_prob = 1.0 - prob.prob_aggressive
                    gross_edge = conservative_prob - poly_no

                fee = compute_polymarket_fee(entry_price, FEE_RATE)
                net_edge = gross_edge - fee

                all_output.append(f"    Raw signal (same-day only):")
                all_output.append(f"      Side: {side}")
                all_output.append(f"      Entry price: {entry_price:.4f}")
                all_output.append(f"      Conservative prob used: {conservative_prob:.4f}")
                all_output.append(f"      Gross edge: {gross_edge:+.4f}")
                all_output.append(f"      Fee ({FEE_RATE}): {fee:.4f}")
                all_output.append(f"      Net edge: {net_edge:+.4f}")

                bin_json["raw_signal"] = {
                    "side": side,
                    "entry_price": round(entry_price, 6),
                    "conservative_prob": round(conservative_prob, 6),
                    "gross_edge": round(gross_edge, 6),
                    "fee": round(fee, 6),
                    "net_edge": round(net_edge, 6),
                }

            # Time-adjusted signal (the new model)
            if adj_prob is not None:
                BASIS_HAIRCUT = 0.02
                NO_NEXT_DAY_HAIRCUT = 0.03
                haircut = BASIS_HAIRCUT + (NO_NEXT_DAY_HAIRCUT if not adj_prob.has_next_day else 0.0)

                all_output.append(f"    Adjusted reference prob ({adj_prob.method}):")
                all_output.append(f"      Conservative: {adj_prob.prob_conservative:.4f}")
                all_output.append(f"      Mid:          {adj_prob.prob_mid:.4f}")
                all_output.append(f"      Aggressive:   {adj_prob.prob_aggressive:.4f}")
                all_output.append(f"      Bounds width: {adj_prob.bounds_width:.4f}")
                all_output.append(f"      dK same={adj_prob.dk_same:.0f}  next={adj_prob.dk_next:.0f}  next_day={'Y' if adj_prob.has_next_day else 'N'}")

                # Evaluate both sides
                adj_yes_gross = adj_prob.prob_conservative - poly_yes
                adj_yes_fee = compute_polymarket_fee(poly_yes, FEE_RATE)
                adj_yes_eff = adj_yes_gross - adj_yes_fee - haircut

                adj_no_gross = (1.0 - adj_prob.prob_aggressive) - poly_no
                adj_no_fee = compute_polymarket_fee(poly_no, FEE_RATE)
                adj_no_eff = adj_no_gross - adj_no_fee - haircut

                if adj_yes_eff >= adj_no_eff:
                    adj_side, adj_entry, adj_gross, adj_fee_v, adj_eff = "BUY_YES", poly_yes, adj_yes_gross, adj_yes_fee, adj_yes_eff
                    alt_side, alt_eff = "BUY_NO", adj_no_eff
                else:
                    adj_side, adj_entry, adj_gross, adj_fee_v, adj_eff = "BUY_NO", poly_no, adj_no_gross, adj_no_fee, adj_no_eff
                    alt_side, alt_eff = "BUY_YES", adj_yes_eff

                all_output.append(f"    Adjusted signal:")
                all_output.append(f"      Best side: {adj_side}  entry={adj_entry:.4f}")
                all_output.append(f"      Gross edge: {adj_gross:+.4f}")
                all_output.append(f"      Fee: {adj_fee_v:.4f}")
                all_output.append(f"      Basis haircut: {haircut:.4f}")
                all_output.append(f"      Effective edge: {adj_eff:+.4f}")
                all_output.append(f"      Alt side: {alt_side}  eff_edge={alt_eff:+.4f}")

                maker_ok = "YES" if adj_eff >= 0.02 else "NO"
                taker_ok = "YES" if adj_eff >= 0.15 else "NO"
                all_output.append(f"      Maker eligible (>=2%): {maker_ok}    Taker eligible (>=15%): {taker_ok}")

                bin_json["adjusted_signal"] = {
                    "method": adj_prob.method,
                    "has_next_day": adj_prob.has_next_day,
                    "prob_conservative": round(adj_prob.prob_conservative, 6),
                    "prob_mid": round(adj_prob.prob_mid, 6),
                    "prob_aggressive": round(adj_prob.prob_aggressive, 6),
                    "bounds_width": round(adj_prob.bounds_width, 6),
                    "side": adj_side,
                    "entry_price": round(adj_entry, 6),
                    "gross_edge": round(adj_gross, 6),
                    "fee": round(adj_fee_v, 6),
                    "haircut": round(haircut, 6),
                    "effective_edge": round(adj_eff, 6),
                    "alt_side": alt_side,
                    "alt_effective_edge": round(alt_eff, 6),
                    "maker_eligible": adj_eff >= 0.02,
                    "taker_eligible": adj_eff >= 0.15,
                }

            all_output.append("")
            event_json["bins"].append(bin_json)

        json_dump["events"][str(exp_date)] = event_json

    # Write text dump
    txt_path = dump_dir / f"leadlag_dump_{timestamp_str}.txt"
    with open(txt_path, "w") as f:
        f.write("\n".join(all_output))
    logger.info(f"Text dump: {txt_path}")

    # Write JSON dump
    json_path = dump_dir / f"leadlag_dump_{timestamp_str}.json"
    with open(json_path, "w") as f:
        json.dump(json_dump, f, indent=2, default=str)
    logger.info(f"JSON dump: {json_path}")

    # Print text to stdout too
    print("\n".join(all_output))


if __name__ == "__main__":
    main()
