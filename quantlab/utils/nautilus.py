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
    """按代码取一个 `Currency`（未注册的代码会被当作加密货币，精度默认 8）。

    改名说明：它以前叫 `get_crypot_currency`——"crypot" 是 "crypto" 的拼写错误
    （2026-09-07 更正）。同一个文件里紧挨着的 `get_crypto_currency_pair` 拼写
    是对的，两个名字并排放着只会让人以为是两类东西。

    同时**去掉了那个 `name: Optional[str] = None` 参数**。它被声明、被接收，
    然后函数体一个字都没用到；两个调用点（`dataset/spot.py` 的
    `base_currency` / `quote_currency`）也都只传 `symbol=`。要真正兑现它，
    得改用 `Currency(code, precision, iso4217, name, currency_type)` 构造器
    ——`Currency.from_str(code, strict=False)` 根本不收 name——那需要替每个
    币种定下 precision / currency_type，仓库里没有任何依据，而且会改变现有两个
    调用点的行为。所以是删，不是补：一个被接收又被忽略的参数，跟这次一并修掉的
    `_train_dl(backtest=...)` 是同一种谎。
    """
    return Currency.from_str(symbol)


def _load_instrument_config(venue: str):
    """Read venue metadata from the packaged instrument file.

    The location comes from `quantlab.utils.paths`, which derives it from
    the package rather than from the current working directory, and is the
    same constant the Binance refresh CLI writes back through.
    """
    with open(INSTRUMENTS_CONFIG_PATH, "r", encoding="utf-8") as f:
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
