#!/usr/bin/env python3
"""
批量获取币安交易规则并更新配置文件
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
    """批量获取并更新交易对配置。

    配置文件的默认位置由 `quantlab.utils.paths` 从包自身推导，不再依赖进程
    的当前工作目录，因此从任何目录运行都会读写同一个随包分发的文件。
    """
    
    # 获取币安交易所信息
    print("正在获取币安交易所信息...")
    exchange_info = _get_binance_exchange_info()

    # 读取现有配置
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {'instruments': {}}
    
    if 'instruments' not in config:
        config['instruments'] = {}
    
    # 如果没有指定symbols，使用配置文件中现有的交易对
    if symbols is None:
        symbols = list(config['instruments'].keys())
    
    
    print(f"正在更新 {len(symbols)} 个交易对的配置...")
    
    # 创建symbol到数据的映射
    symbol_map = {s['symbol']: s for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
    
    updated_count = 0
    for symbol in symbols:
        if symbol in symbol_map:
            symbol_info = _parse_symbol_info(symbol_map[symbol])
            config['instruments'][symbol] = symbol_info
            print(f"✓ 已更新 {symbol}")
            updated_count += 1
        else:
            print(f"✗ 未找到交易对: {symbol}")
    

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
    
    # 保存配置
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, 
                  indent=2, sort_keys=False)
    
    print(f"\n✅ 成功更新了 {updated_count} 个交易对的配置")
    print(f"配置文件已保存到: {config_file.absolute()}")


def get_all_usdt_pairs(limit: int = 50):
    """获取所有USDT交易对（按24h交易量排序）"""
    print("正在获取币安交易所信息...")
    exchange_info = _get_binance_exchange_info()

    # 获取24h统计数据用于排序
    ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(ticker_url, timeout=10)
        response.raise_for_status()
        ticker_data = response.json()
        
        # 创建交易量映射
        volume_map = {t['symbol']: float(t['quoteVolume']) for t in ticker_data}
        
    except requests.RequestException as e:
        print(f"获取24h统计数据失败: {e}")
        volume_map = {}
    
    # 筛选USDT交易对并按交易量排序
    usdt_pairs = []
    for symbol_data in exchange_info['symbols']:
        symbol = symbol_data['symbol']
        if (symbol.endswith('USDT') and 
            symbol_data['status'] == 'TRADING' and
            symbol_data['isSpotTradingAllowed']):
            
            volume = volume_map.get(symbol, 0)
            usdt_pairs.append((symbol, volume))
    
    # 按交易量排序并取前N个
    usdt_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [pair[0] for pair in usdt_pairs[:limit]]
    
    print(f"找到 {len(usdt_pairs)} 个USDT交易对，选择前 {limit} 个（按24h交易量排序）:")
    for i, pair in enumerate(top_pairs[:10], 1):
        volume = volume_map.get(pair, 0)
        print(f"{i:2d}. {pair:<12} (24h成交量: ${volume:,.0f})")
    
    if len(top_pairs) > 10:
        print(f"... 还有 {len(top_pairs) - 10} 个交易对")
    
    return top_pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="获取币安交易规则并更新配置")
    parser.add_argument('--symbols', '-s', nargs='+', 
                       help='指定要更新的交易对，如: BTCUSDT ETHUSDT')
    parser.add_argument('--config', '-c', default=INSTRUMENTS_CONFIG_PATH,
                       help='配置文件路径（默认为随包分发的实例元数据文件）')
    parser.add_argument('--top-usdt', '-t', type=int, metavar='N',
                       help='获取前N个USDT交易对（按24h交易量排序）')
    parser.add_argument('--list-top', '-l', type=int, metavar='N',
                       help='仅列出前N个USDT交易对，不更新配置')
    
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
        print(f"❌ 错误: {e}")
        exit(1)