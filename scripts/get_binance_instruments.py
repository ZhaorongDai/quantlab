#!/usr/bin/env python3
"""Refresh the Binance spot instrument metadata file from the live exchange.

The script downloads Binance's public ``exchangeInfo`` payload, which lists
every symbol's trading rules (tick size, lot size, minimum order value and so
on). It flattens the rules of the requested symbols with
``quantlab.utils.binance`` and writes them into the packaged
``config/instruments.yaml``, or into the file given with ``--config``. The
Nautilus helpers read that file when they build instrument definitions. The script
can also rank the USDT pairs by 24-hour quote volume and either list the top
N or refresh the file with them. No credentials are needed, because both
Binance endpoints are public.

Usage::

    uv run python scripts/get_binance_instruments.py --help

    # Refresh every symbol already present in the file.
    uv run python scripts/get_binance_instruments.py

    # Refresh specific symbols.
    uv run python scripts/get_binance_instruments.py --symbols BTCUSDT ETHUSDT

    # Refresh the file with the top 20 USDT pairs by 24h volume.
    uv run python scripts/get_binance_instruments.py --top-usdt 20

    # Only list the top 20 USDT pairs; write nothing.
    uv run python scripts/get_binance_instruments.py --list-top 20

    # Write to another file.
    uv run python scripts/get_binance_instruments.py --config ./instruments.yaml
"""

import json
import requests
import yaml
from typing import Dict, Any
from pathlib import Path

from quantlab.utils.binance import _get_binance_exchange_info, _parse_symbol_info
from quantlab.utils.paths import INSTRUMENTS_CONFIG_PATH


def update_instruments_config(
    symbols: list = None, config_path: str = INSTRUMENTS_CONFIG_PATH
):
    """Fetch the trading rules of ``symbols`` and write them into the YAML file.

    Symbols that are not currently trading on Binance spot are reported and
    skipped. Default ``fees`` and ``margin`` sections are added when the file
    has none. The default file location is derived from the package by
    ``quantlab.utils.paths``, so the same packaged file is read and written
    whatever the current working directory is.

    Parameters
    ----------
    symbols : list of str, optional
        Symbols to refresh. ``None`` (the default) refreshes every symbol
        already present in the file.
    config_path : str, default ``INSTRUMENTS_CONFIG_PATH``
        The YAML file to update. It is created if missing.

    Examples
    --------
    Needs network access to the Binance API::

        update_instruments_config(["BTCUSDT", "ETHUSDT"])
        update_instruments_config(config_path="./instruments.yaml")
    """

    print("Fetching Binance exchange info...")
    exchange_info = _get_binance_exchange_info()

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {'instruments': {}}

    if 'instruments' not in config:
        config['instruments'] = {}

    # Without an explicit list, refresh the symbols the file already holds.
    if symbols is None:
        symbols = list(config['instruments'].keys())


    print(f"Updating {len(symbols)} instrument(s)...")

    symbol_map = {s['symbol']: s for s in exchange_info['symbols'] if s['status'] == 'TRADING'}

    updated_count = 0
    for symbol in symbols:
        if symbol in symbol_map:
            symbol_info = _parse_symbol_info(symbol_map[symbol])
            config['instruments'][symbol] = symbol_info
            print(f"✓ Updated {symbol}")
            updated_count += 1
        else:
            print(f"✗ Symbol not found: {symbol}")


    if 'fees' not in config:
        config['fees'] = {
            'maker_fee': 0.001,
            'taker_fee': 0.001
        }

    if 'margin' not in config:
        config['margin'] = {
            'margin_init': 0,
            'margin_maint': 0
        }

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  indent=2, sort_keys=False)

    print(f"\n✅ Updated {updated_count} instrument(s)")
    print(f"Config written to: {config_file.absolute()}")


def get_all_usdt_pairs(limit: int = 50):
    """Return the top ``limit`` spot USDT pairs ranked by 24-hour quote volume.

    Only symbols that are trading and allowed for spot trading are
    considered. If the 24-hour ticker request fails, every pair is ranked
    with volume 0 and the order is Binance's own. The first ten are printed.

    Parameters
    ----------
    limit : int, default 50
        How many pairs to return.

    Returns
    -------
    list[str]
        A list of symbol strings, highest volume first.

    Examples
    --------
    Needs network access to the Binance API::

        top = get_all_usdt_pairs(limit=20)
        update_instruments_config(top)
    """
    print("Fetching Binance exchange info...")
    exchange_info = _get_binance_exchange_info()

    ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(ticker_url, timeout=10)
        response.raise_for_status()
        ticker_data = response.json()

        volume_map = {t['symbol']: float(t['quoteVolume']) for t in ticker_data}

    except requests.RequestException as e:
        print(f"Failed to fetch 24h ticker statistics: {e}")
        volume_map = {}

    usdt_pairs = []
    for symbol_data in exchange_info['symbols']:
        symbol = symbol_data['symbol']
        if (symbol.endswith('USDT') and
            symbol_data['status'] == 'TRADING' and
            symbol_data['isSpotTradingAllowed']):

            volume = volume_map.get(symbol, 0)
            usdt_pairs.append((symbol, volume))

    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [pair[0] for pair in usdt_pairs[:limit]]

    print(
        f"Found {len(usdt_pairs)} USDT pairs; selecting the top {limit} "
        f"by 24h volume:"
    )
    for i, pair in enumerate(top_pairs[:10], 1):
        volume = volume_map.get(pair, 0)
        print(f"{i:2d}. {pair:<12} (24h volume: ${volume:,.0f})")

    if len(top_pairs) > 10:
        print(f"... and {len(top_pairs) - 10} more")

    return top_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Binance trading rules and update the instrument config"
    )
    parser.add_argument('--symbols', '-s', nargs='+',
                       help='Symbols to refresh, e.g. BTCUSDT ETHUSDT')
    parser.add_argument('--config', '-c', default=INSTRUMENTS_CONFIG_PATH,
                       help='Config file path (defaults to the packaged '
                            'instrument metadata file)')
    parser.add_argument('--top-usdt', '-t', type=int, metavar='N',
                       help='Refresh the top N USDT pairs by 24h volume')
    parser.add_argument('--list-top', '-l', type=int, metavar='N',
                       help='Only list the top N USDT pairs; do not update the config')

    args = parser.parse_args()

    try:
        if args.list_top:
            get_all_usdt_pairs(args.list_top)
        elif args.top_usdt:
            symbols = get_all_usdt_pairs(args.top_usdt)
            update_instruments_config(symbols, args.config)
        else:
            update_instruments_config(args.symbols, args.config)

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
