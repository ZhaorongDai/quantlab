"""Fetch Binance spot instrument metadata from the public exchange-info endpoint.

``get_instrument_info`` turns Binance's ``exchangeInfo`` response into the flat
per-symbol dictionary stored under ``instruments`` in the packaged
``instruments.yaml``, which the Nautilus helpers read when building a
``CurrencyPair``. It is used both by the instrument refresh CLI and as a
fallback when a symbol is missing from the packaged file.
"""

from typing import Any, Dict

import requests
from loguru import logger


def get_instrument_info(symbols: list[str]) -> dict:
    """Return ``{"instruments": {symbol: info}}`` for the requested Binance symbols.

    One request is made for the whole exchange and the requested symbols are
    picked out of it. A symbol Binance does not list is logged as a warning and
    omitted from the result rather than raising.

    Parameters
    ----------
    symbols : list[str]
        Binance spot symbols such as ``"BTCUSDT"``.

    Returns
    -------
    dict
        A dict with a single ``"instruments"`` key mapping each found symbol to
        the precision, increment, quantity, price and notional limits produced
        by ``_parse_symbol_info``.

    Raises
    ------
    requests.RequestException
        If the exchange-info request fails.

    Examples
    --------
    Needs network access to ``api.binance.com``:

    >>> info = get_instrument_info(["BTCUSDT", "ETHUSDT"])
    >>> sorted(info["instruments"])
    ['BTCUSDT', 'ETHUSDT']
    >>> sorted(info["instruments"]["BTCUSDT"])[:3]
    ['max_price', 'max_quantity', 'min_notional']
    """
    exchange_info = _get_binance_exchange_info()
    symbol_map = {s["symbol"]: s for s in exchange_info["symbols"]}
    config = {
        "instruments": {},
    }
    for symbol in symbols:
        if symbol in symbol_map:
            symbol_info = _parse_symbol_info(symbol_map[symbol])
            config["instruments"][symbol] = symbol_info
            logger.info(f"Updated {symbol} info")
        else:
            logger.warning(f"Symbol not found: {symbol}")
    return config


def _get_binance_exchange_info() -> Dict[str, Any]:
    """Download and return the full Binance spot ``exchangeInfo`` payload."""
    url = "https://api.binance.com/api/v3/exchangeInfo"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to get Binance exchange info: {e}")
        raise e


def _parse_symbol_info(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one ``exchangeInfo`` symbol entry into the instrument config shape.

    Precisions come from the symbol record; increments and quantity or price
    bounds come from the ``PRICE_FILTER`` and ``LOT_SIZE`` filters; the minimum
    notional is read from ``MIN_NOTIONAL`` when present and ``NOTIONAL``
    otherwise. Missing filters fall back to conservative defaults.
    """
    filters = {f["filterType"]: f for f in symbol_data["filters"]}

    price_filter = filters.get("PRICE_FILTER", {})

    lot_size = filters.get("LOT_SIZE", {})

    min_notional = filters.get("MIN_NOTIONAL", {})
    notional_filter = filters.get("NOTIONAL", {})

    min_notional_value = 0.0
    if min_notional:
        min_notional_value = float(min_notional.get("minNotional", 0))
    elif notional_filter:
        min_notional_value = float(notional_filter.get("minNotional", 0))

    return {
        "price_precision": symbol_data.get("quotePrecision", 2),
        "size_precision": symbol_data.get("baseAssetPrecision", 6),
        "price_increment": float(price_filter.get("tickSize", 0.01)),
        "size_increment": float(lot_size.get("stepSize", 0.000001)),
        "min_quantity": float(lot_size.get("minQty", 0.000001)),
        "max_quantity": float(lot_size.get("maxQty", 9000)),
        "min_price": float(price_filter.get("minPrice", 0.01)),
        "max_price": float(price_filter.get("maxPrice", 1000000)),
        "min_notional": min_notional_value,
    }
