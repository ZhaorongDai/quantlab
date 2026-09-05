import os
from decimal import Decimal
from typing import Optional

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


def get_crypot_currency(symbol: str, name: Optional[str] = None):
    return Currency.from_str(symbol)


def _load_instrument_config(venue: str):
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "instruments.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)[venue]


def get_crypto_currency_pair(
    symbol: str,
    venue: str,
    base: Currency,
    quote: Currency,
):
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
    """生成bar type字符串"""
    time_interval_minutes = time_interval.astype("timedelta64[m]").astype(
        "int64"
    )

    # 特殊情况：1天、1小时、1分钟
    if time_interval == np.timedelta64(1, "D"):
        return f"{symbol}.{venue}-1-DAY-LAST-EXTERNAL"
    elif time_interval == np.timedelta64(1, "h"):
        return f"{symbol}.{venue}-1-HOUR-LAST-EXTERNAL"
    elif time_interval == np.timedelta64(1, "m"):
        return f"{symbol}.{venue}-1-MINUTE-LAST-EXTERNAL"

    # 其他情况：根据分钟数计算
    if time_interval_minutes % (60 * 24) == 0:
        days = time_interval_minutes // (60 * 24)
        return f"{symbol}.{venue}-{days}-DAY-LAST-EXTERNAL"
    elif time_interval_minutes % 60 == 0:
        hours = time_interval_minutes // 60
        return f"{symbol}.{venue}-{hours}-HOUR-LAST-EXTERNAL"
    else:
        return f"{symbol}.{venue}-{time_interval_minutes}-MINUTE-LAST-EXTERNAL"


def parse_symbol_currencies(symbol: str) -> tuple[str, str]:
    """解析symbol获取base和quote货币"""
    if symbol.endswith("USDT"):
        base_symbol = symbol[:-4]  # 移除USDT
        quote_symbol = "USDT"
    else:
        raise ValueError(f"Unsupported symbol format: {symbol}")

    return base_symbol, quote_symbol
