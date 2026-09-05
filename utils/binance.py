from typing import Any, Dict

import requests
from loguru import logger


def get_instrument_info(symbols: list[str]) -> dict:
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
    """获取币安交易所信息"""
    url = "https://api.binance.com/api/v3/exchangeInfo"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to get Binance exchange info: {e}")
        raise e


def _parse_symbol_info(symbol_data: Dict[str, Any]) -> Dict[str, Any]:
    """解析单个交易对信息"""
    filters = {f["filterType"]: f for f in symbol_data["filters"]}

    # 价格过滤器
    price_filter = filters.get("PRICE_FILTER", {})

    # 数量过滤器
    lot_size = filters.get("LOT_SIZE", {})

    # 最小名义价值过滤器
    min_notional = filters.get("MIN_NOTIONAL", {})
    notional_filter = filters.get("NOTIONAL", {})

    # 获取最小名义价值
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
