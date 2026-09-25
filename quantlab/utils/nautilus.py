"""Helpers for converting quantlab market data into Nautilus Trader objects.

Nautilus Trader is an event-driven trading framework; quantlab currently uses
only its data model and its parquet data catalog. The spot kline dataset uses
these helpers to build ``Currency`` and ``CurrencyPair`` instruments for its
catalog and to name bar types. A *venue* is the exchange an instrument trades
on, such as ``"BINANCE"``. Instrument limits are read from the packaged
``instruments.yaml``; a symbol missing from that file is fetched live from
the venue (currently only Binance).
"""

import os
from decimal import Decimal

import numpy as np
import yaml
from loguru import logger
from nautilus_trader.model.identifiers import (
    InstrumentId,
    Symbol,
    Venue,
)
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from .binance import get_instrument_info as get_instrument_info_binance
from quantlab.utils.paths import INSTRUMENTS_CONFIG_PATH


def get_crypto_currency(symbol: str) -> Currency:
    """Return the Nautilus ``Currency`` for ``symbol``.

    Codes Nautilus does not know are registered on the fly as crypto
    currencies with the default precision of 8.

    Parameters
    ----------
    symbol : str
        A currency code such as ``"BTC"`` or ``"USDT"``.

    Returns
    -------
    Currency
        The matching Nautilus currency.

    Examples
    --------
    >>> btc = get_crypto_currency("BTC")
    >>> btc.code, btc.precision
    ('BTC', 8)
    """
    return Currency.from_str(symbol)


def _load_instrument_config(venue: str):
    """Return the ``venue`` section of the packaged ``instruments.yaml``."""
    with open(INSTRUMENTS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[venue]


def get_crypto_currency_pair(
    symbol: str,
    venue: str,
    base: Currency,
    quote: Currency,
):
    """Build a Nautilus ``CurrencyPair`` instrument for ``symbol`` on ``venue``.

    Precision, increment, quantity, price and notional limits come from the
    packaged instrument file. Fees and margins come from the venue-level
    ``fees`` and ``margin`` sections. A symbol absent from the file is fetched
    from the exchange for this call only, without updating the file.

    Parameters
    ----------
    symbol : str
        The venue's symbol, such as ``"BTCUSDT"``.
    venue : str
        The venue name as spelled in ``instruments.yaml``.
    base : Currency
        The base currency.
    quote : Currency
        The quote currency.

    Returns
    -------
    CurrencyPair
        A ``CurrencyPair`` with ``ts_event`` and ``ts_init`` set to 0.

    Raises
    ------
    ValueError
        If the symbol is missing from the file and the venue has no live
        fetcher.

    Examples
    --------
    >>> base, quote = get_crypto_currency("BTC"), get_crypto_currency("USDT")
    >>> pair = get_crypto_currency_pair("BTCUSDT", "BINANCE", base, quote)
    >>> str(pair.id), pair.price_precision, pair.size_precision
    ('BTCUSDT.BINANCE', 8, 8)

    A symbol absent from the packaged file is fetched from the venue, so
    that path needs network access.
    """
    config_data = _load_instrument_config(venue)

    if symbol not in config_data["instruments"]:
        logger.warning(
            f"Symbol {symbol} not found in config, fetching from exchange {venue}"
        )
        match venue:
            case "BINANCE":
                new_data = get_instrument_info_binance([symbol])
            case _:
                raise ValueError(f"Unsupported venue: {venue}")
        config_data.update(new_data)

    instrument_config = config_data["instruments"][symbol]
    fees_config = config_data["fees"]
    margin_config = config_data["margin"]

    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol(symbol),
            venue=Venue(venue),
        ),
        raw_symbol=Symbol(symbol),
        base_currency=base,
        quote_currency=quote,
        price_precision=instrument_config["price_precision"],
        size_precision=instrument_config["size_precision"],
        price_increment=Price(
            instrument_config["price_increment"],
            precision=instrument_config["price_precision"],
        ),
        size_increment=Quantity(
            instrument_config["size_increment"],
            precision=instrument_config["size_precision"],
        ),
        lot_size=None,
        max_quantity=Quantity(
            instrument_config["max_quantity"],
            precision=instrument_config["size_precision"],
        ),
        min_quantity=Quantity(
            instrument_config["min_quantity"],
            precision=instrument_config["size_precision"],
        ),
        max_notional=None,
        min_notional=Money(instrument_config["min_notional"], quote),
        max_price=Price(
            instrument_config["max_price"],
            precision=instrument_config["price_precision"],
        ),
        min_price=Price(
            instrument_config["min_price"],
            precision=instrument_config["price_precision"],
        ),
        margin_init=Decimal(str(margin_config["margin_init"])),
        margin_maint=Decimal(str(margin_config["margin_maint"])),
        maker_fee=Decimal(str(fees_config["maker_fee"])),
        taker_fee=Decimal(str(fees_config["taker_fee"])),
        ts_event=0,
        ts_init=0,
    )


def generate_bar_type_str(
    time_interval: np.timedelta64, symbol: str, venue: str = "BINANCE"
) -> str:
    """Return the Nautilus bar-type string for a bar of ``time_interval``.

    The interval is expressed in the largest unit that divides it exactly:
    whole days as ``DAY``, whole hours as ``HOUR``, anything else in minutes.
    Bars are always ``LAST`` priced (built from trade prices) and
    ``EXTERNAL`` aggregated (built by the data vendor, not by Nautilus).

    Parameters
    ----------
    time_interval : np.timedelta64
        Length of one bar.
    symbol : str
        The venue's symbol, such as ``"BTCUSDT"``.
    venue : str, default "BINANCE"
        The venue name.

    Returns
    -------
    str
        A string Nautilus's ``BarType.from_str`` accepts.

    Examples
    --------
    >>> generate_bar_type_str(np.timedelta64(4, "h"), "BTCUSDT")
    'BTCUSDT.BINANCE-4-HOUR-LAST-EXTERNAL'
    """
    time_interval_minutes = time_interval.astype("timedelta64[m]").astype(
        "int64"
    )

    if time_interval == np.timedelta64(1, "D"):
        return f"{symbol}.{venue}-1-DAY-LAST-EXTERNAL"
    elif time_interval == np.timedelta64(1, "h"):
        return f"{symbol}.{venue}-1-HOUR-LAST-EXTERNAL"
    elif time_interval == np.timedelta64(1, "m"):
        return f"{symbol}.{venue}-1-MINUTE-LAST-EXTERNAL"

    if time_interval_minutes % (60 * 24) == 0:
        days = time_interval_minutes // (60 * 24)
        return f"{symbol}.{venue}-{days}-DAY-LAST-EXTERNAL"
    elif time_interval_minutes % 60 == 0:
        hours = time_interval_minutes // 60
        return f"{symbol}.{venue}-{hours}-HOUR-LAST-EXTERNAL"
    else:
        return f"{symbol}.{venue}-{time_interval_minutes}-MINUTE-LAST-EXTERNAL"


def parse_symbol_currencies(symbol: str) -> tuple[str, str]:
    """Split a ``...USDT`` spot symbol into its ``(base, quote)`` codes.

    Only USDT-quoted symbols are recognised.

    Parameters
    ----------
    symbol : str
        A spot symbol such as ``"ETHUSDT"``.

    Returns
    -------
    tuple[str, str]
        The base and quote currency codes.

    Raises
    ------
    ValueError
        If ``symbol`` does not end in ``USDT``.

    Examples
    --------
    >>> parse_symbol_currencies("ETHUSDT")
    ('ETH', 'USDT')
    """
    if symbol.endswith("USDT"):
        base_symbol = symbol[:-4]
        quote_symbol = "USDT"
    else:
        raise ValueError(f"Unsupported symbol format: {symbol}")

    return base_symbol, quote_symbol
