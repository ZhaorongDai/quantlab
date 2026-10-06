"""PROTOTYPE, throwaway: the B (Dashboard) page with the F2 factor attribution, in English or Chinese.

Questions:
- does a language switch (English default, Chinese) work for the whole page;
- does an explanation on hovering every metric help, in the page's language;
- do the "by part" charts read better as plain bars from the zero axis
  (the waterfall's floating bars were unclear)?

Everything is on one page: ``report_f2_i18n.html``. The EN / 中文 switch is in
the header (remembered in the browser); hovering a KPI card, a table row, a
factor-attribution tile or the ⓘ beside a chart title shows its explanation.

Run: ``uv run python quantlab/runs/prototype_report_f2_i18n.py ARGS.pkl OUT_DIR``
"""

import html
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantlab.dataset._support.ff48 import FF48_INDUSTRIES  # PROTOTYPE only: production needs names in the store
from quantlab.runs import backtest_report as br
from quantlab.runs import prototype_report_variants as pv

BARS_PER_YEAR = 252
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG, MID, TOTAL = "#2a78d6", "#e34948", "#f0efec", "#52514e"
PARTS = [
    ("country", "#2a78d6"), ("industry", "#eb6834"), ("style", "#1baf7a"), ("specific", "#eda100"),
    ("uncovered", "#e87ba4"), ("risk_free", "#008300"), ("trading", "#4a3aa7"),
]

# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------

#: Chart and page words: key -> (English, Chinese).
W = {
    "country": ("Country", "国家（市场）"), "industry": ("Industry", "行业"), "style": ("Style", "风格"),
    "specific": ("Specific", "特异（选股）"), "uncovered": ("Uncovered", "未覆盖"),
    "risk_free": ("Risk-free", "无风险利率"), "trading": ("Trading", "交易"), "total": ("Total", "合计"),
    "forecast": ("Forecast", "预测"), "realized": ("Realized", "实现"),
    "a_year": ("a year", "/年"), "log_nav": ("Total (log NAV)", "合计（对数净值）"),
    "ann_vol": ("annualized volatility", "年化波动率"),
    "mean_exp": ("Mean net exposure", "平均净暴露"), "contrib": ("Contribution, log growth / yr", "贡献（年化对数增长）"),
    "fc_risk": ("Forecast (x-sigma-rho)", "预测（x-sigma-rho）"),
    "rz_risk": ("Realized, cov(c, r) / sigma(r)", "实现，cov(c, r) / sigma(r)"),
    "ret_bar": ("Return, log growth / yr", "收益（年化对数增长）"),
    "risk_bar": ("Realized risk contribution", "实现风险贡献"),
    "realized63": ("Realized (63-bar)", "实现（63 根 bar）"),
    "exposure": ("exposure", "暴露"), "week": ("week of", "当周"),
    "of_fc_vol": ("of forecast vol", "的预测波动率"), "mean_weight": ("mean net weight", "平均净权重"),
}

STYLE_ZH = {
    "size": "规模", "beta": "贝塔", "momentum": "动量", "residual volatility": "残差波动率",
    "nonlinear size": "非线性规模", "nonlinear beta": "非线性贝塔", "liquidity": "流动性",
    "dividend yield": "股息率", "book to price": "账面市值比", "earnings yield": "盈利收益率",
    "leverage": "杠杆", "growth": "成长",
}
INDUSTRY_ZH = {
    1: "农业", 2: "食品", 3: "糖果与软饮", 4: "啤酒与烈酒", 5: "烟草", 6: "休闲娱乐用品", 7: "娱乐",
    8: "印刷出版", 9: "消费品", 10: "服装", 11: "医疗服务", 12: "医疗器械", 13: "制药", 14: "化工",
    15: "橡胶塑料", 16: "纺织", 17: "建材", 18: "建筑", 19: "钢铁", 20: "金属制品", 21: "机械",
    22: "电气设备", 23: "汽车与卡车", 24: "航空器", 25: "船舶与铁路设备", 26: "国防", 27: "贵金属",
    28: "非金属与工业金属采矿", 29: "煤炭", 30: "石油天然气", 31: "公用事业", 32: "通信", 33: "个人服务",
    34: "商业服务", 35: "计算机", 36: "电子设备", 37: "测量与控制设备", 38: "商业用品", 39: "包装容器",
    40: "交通运输", 41: "批发", 42: "零售", 43: "餐饮酒店", 44: "银行", 45: "保险", 46: "房地产",
    47: "金融交易", 48: "其他",
}
INDUSTRY_EN = {f"industry_{i.code}": i.name for i in FF48_INDUSTRIES}


def name_of(factor: str, lang: str) -> str:
    if factor in INDUSTRY_EN:
        return INDUSTRY_EN[factor] if lang == "en" else INDUSTRY_ZH[int(factor.split("_")[1])]
    plain = re.sub(r"^style_", "", factor).replace("_", " ")
    return plain if lang == "en" else STYLE_ZH.get(plain, plain)


def w(key: str, lang: str) -> str:
    return W[key][0 if lang == "en" else 1]


#: Page text the browser swaps on the language switch: English -> Chinese.
ZH = {
    # navigation and headings
    "Overview": "概览", "Excess": "超额", "Rolling": "滚动", "Portfolio": "组合", "Attribution": "归因",
    "Factor attribution": "因子归因", "Setup & notes": "设置与说明", "Windows": "时间窗口", "Setup": "设置",
    "Notes": "说明", "Trading": "交易", "In-sample vs out-of-sample": "样本内 vs 样本外",
    "Monthly returns by year": "按年按月收益", "Strategy": "策略", "Difference": "差值", "Value": "数值",
    "In-sample": "样本内", "Out-of-sample": "样本外", "Whole": "全区间", "Returns": "收益", "Risk": "风险",
    "Risk-adjusted": "风险调整后", "Win rates": "胜率", "Return": "收益", "model": "模型", "training": "训练",
    "traded, out-of-sample": "交易，样本外", "traded, in-sample": "交易，样本内",
    # KPI cards
    "Total return": "总收益", "Excess return": "超额收益", "Information ratio": "信息比率", "Win rate": "胜率",
    "Sharpe ratio": "夏普比率", "Max drawdown": "最大回撤", "Beta": "贝塔", "Turnover / year": "年换手",
    "Annualised return": "年化收益", "Volatility": "波动率",
    # strategy metrics
    "Start value": "期初净值", "End value": "期末净值", "Annualised volatility": "年化波动率",
    "Longest drawdown": "最长回撤期", "Value at risk (95%)": "在险价值（95%）", "Skew": "偏度", "Kurtosis": "峰度",
    "Sortino ratio": "索提诺比率", "Calmar ratio": "卡玛比率", "Omega ratio": "欧米茄比率", "Tail ratio": "尾部比率",
    "Common sense ratio": "常识比率", "Rebalances with a gain": "盈利的调仓期占比", "Months with a gain": "盈利月份占比",
    # trading
    "Annualised turnover": "年化换手", "Turnover per rebalance": "每次调仓换手", "Total turnover": "总换手",
    "Traded notional": "成交金额", "Fees paid": "已付费用", "Max gross exposure": "最大总敞口",
    "Orders filled": "成交笔数", "Round trips": "往返交易", "Round trips closed": "已平仓往返",
    "Round trips open": "未平仓往返", "Open round-trip P&L": "未平仓往返盈亏", "Round-trip win rate": "往返胜率",
    "Best round trip": "最佳往返", "Worst round trip": "最差往返", "Avg winning round trip": "平均盈利往返",
    "Avg losing round trip": "平均亏损往返", "Avg winning round-trip duration": "盈利往返平均时长",
    "Avg losing round-trip duration": "亏损往返平均时长", "Profit factor": "盈亏因子", "Expectancy": "期望收益",
    "Orders rejected": "被拒订单", "Rebalances held after a failure": "因构建失败而保持持仓的调仓",
    # relative
    "Excess return (geometric)": "超额收益（几何）", "Annualised excess return": "年化超额收益",
    "Total return difference (arithmetic)": "总收益差（算术）", "Excess max drawdown": "超额最大回撤",
    "Tracking error": "跟踪误差", "Correlation": "相关系数", "CAPM alpha": "CAPM 阿尔法",
    "Bars beating the benchmark": "跑赢基准的 bar 占比", "Rebalances beating the benchmark": "跑赢基准的调仓期占比",
    "Months beating the benchmark": "跑赢基准的月份占比",
    # setup
    "Bar interval": "Bar 间隔", "Benchmark": "基准", "Deepest drawdown (valley to recovery)": "最深回撤（谷底到恢复）",
    "Model mode": "模型模式", "Rebalance every": "调仓间隔", "Portfolio construction": "组合构建", "Fees": "费率",
    # notes
    "No borrow or short-financing cost is modelled, so short-side returns are optimistic.":
        "未计入融券或空头融资成本，因此空头一侧的收益偏乐观。",
    "The trade metrics are the position level view: one entry to flat round trip per symbol, so a partial trim of a holding is not counted as its own closed trade. Counting every trim as a closed trade is what vectorbt does by default, and it inflates the win rate. The row named Total Orders is the number of fills that actually happened over the window.":
        "交易指标按持仓口径统计：每个标的从建仓到清仓算一次往返，部分减仓不单独算一笔平仓。vectorbt 默认把每次减仓都算成平仓，这会抬高胜率。“成交笔数”一行是窗口内实际发生的成交次数。",
    "The two triangles on the equity curve mark the deepest drawdown: the up triangle is its deepest bar, that is its valley, and the down triangle is the bar it recovered. The distance between them is how long it took to get from the bottom back to even, counted in trading days, that is in bars, never in calendar days. It is not the metric named Max Drawdown Duration, which measures the longest drawdown and counts from where that drawdown began, so the two numbers usually differ.":
        "净值曲线上的两个三角标出最深的一次回撤：朝上的三角是谷底，朝下的三角是恢复到前高的那根 bar。两者之间的距离是从谷底回到前高所用的时间，按交易日（bar）计，不按日历日。它不同于“最长回撤期”：后者衡量持续最久的一次回撤，并从回撤开始时算起，所以两个数字通常不同。",
}

#: Regex rules for text carrying run values: (pattern, Chinese replacement).
ZH_RULES = [
    (r"^Strategy vs (.+)$", r"策略 vs \1"), (r"^Relative to (.+?)( \(out-of-sample\))?$", r"相对 \1"),
    (r"^Trading \(out-of-sample\)$", "交易（样本外）"), (r"^annualised (.+)$", r"年化 \1"),
    (r"^tracking error (.+)$", r"跟踪误差 \1"), (r"^monthly (.+)$", r"按月 \1"),
    (r"^correlation (.+)$", r"相关系数 \1"), (r"^fees (.+)$", r"费用 \1"), (r"^longest (.+)$", r"最长 \1"),
    (r"^Sortino (.+)$", r"索提诺 \1"), (r"^end value (.+)$", r"期末净值 \1"),
    (r"^Backtest (.+) \.\. (.+) \((.+) bars\)$", r"回测 \1 .. \2（\3 根 bar）"),
    (r"^(\S+) → (\S+)\s+· vs (.+)$", r"\1 → \2 · 对比 \3"),
]

#: Explanations shown on hover: English label -> (English, Chinese). Plain words first, the formula after.
TIPS = {
    # KPI cards and metric rows
    "Total return": ("How much the portfolio grew over the whole window, all compounding included. 42% means 1.00 became 1.42.",
                     "整个区间内组合增长了多少（含复利）。42% 表示 1.00 变成了 1.42。"),
    "Annualised return": ("The yearly growth rate that would produce the same total return: the total return spread evenly over the years.",
                          "把总收益平摊到每一年后的年增长率：按这个速度每年复利，正好得到同样的总收益。"),
    "Excess return": ("How far the portfolio ended ahead of (or behind) the benchmark: strategy value / benchmark value - 1 at the last bar.",
                      "组合最终领先（或落后）基准多少：最后一根 bar 上 策略净值 / 基准净值 - 1。"),
    "Excess return (geometric)": ("How far the portfolio ended ahead of (or behind) the benchmark: strategy value / benchmark value - 1 at the last bar.",
                                  "组合最终领先（或落后）基准多少：最后一根 bar 上 策略净值 / 基准净值 - 1。"),
    "Annualised excess return": ("The excess return over the benchmark expressed per year.",
                                 "相对基准的超额收益，换算成每年。"),
    "Total return difference (arithmetic)": ("Strategy total return minus benchmark total return. Differs from the geometric excess because of compounding.",
                                             "策略总收益减去基准总收益。由于复利，它和几何超额收益不同。"),
    "Information ratio": ("Excess return per unit of tracking error: how consistently the portfolio beat the benchmark. Above 0.5 is good, negative means it lagged.",
                          "每单位跟踪误差换来的超额收益，衡量跑赢基准是否稳定。高于 0.5 算好，为负说明落后于基准。"),
    "Tracking error": ("How much the portfolio's returns wander from the benchmark's, per year: the volatility of the daily difference.",
                       "组合收益偏离基准收益的程度（年化）：每日收益差的波动率。"),
    "Win rate": ("Share of rebalance periods in which the portfolio did better than the benchmark (or made money, without a benchmark).",
                 "调仓期中组合跑赢基准（没有基准时为盈利）的比例。"),
    "Sharpe ratio": ("Return per unit of risk: annualized mean return divided by annualized volatility (no risk-free rate subtracted). Higher is better; 1 is good.",
                     "每单位风险换来的收益：年化平均收益除以年化波动率（未扣无风险利率）。越高越好，1 算不错。"),
    "Max drawdown": ("The worst fall from a previous high to a later low, in percent. It is how much you would have lost buying at the worst time.",
                     "从前一个高点到之后低点的最大跌幅（百分比），相当于在最差时点买入会亏多少。"),
    "Beta": ("How much the portfolio moves when the benchmark moves 1%. 0.65 means it tends to move 0.65% for every 1% of the benchmark.",
             "基准涨跌 1% 时组合通常涨跌多少。0.65 表示基准每动 1%，组合大约动 0.65%。"),
    "Correlation": ("How closely the portfolio's daily returns move with the benchmark's, from -1 (opposite) to 1 (in lockstep).",
                    "组合与基准日收益同步的程度，从 -1（完全相反）到 1（完全同步）。"),
    "Turnover / year": ("How much of the portfolio is bought and sold in a year. 700% means the book is replaced about 3.5 times a year (each replacement is a sell plus a buy).",
                        "一年内买卖的金额占组合的比例。700% 大约相当于一年把整个组合换掉 3.5 次（每次换仓包括卖出和买入）。"),
    "Volatility": ("How much the daily returns swing, scaled to a year. About two thirds of years land within ± this much of the average.",
                   "日收益的波动幅度，换算成一年。大约三分之二的年份落在平均值 ± 这个幅度之内。"),
    "Annualised volatility": ("How much the daily returns swing, scaled to a year. About two thirds of years land within ± this much of the average.",
                              "日收益的波动幅度，换算成一年。大约三分之二的年份落在平均值 ± 这个幅度之内。"),
    "Start value": ("Portfolio value at the first bar.", "第一根 bar 的组合净值。"),
    "End value": ("Portfolio value at the last bar.", "最后一根 bar 的组合净值。"),
    "Longest drawdown": ("The longest time the portfolio spent below a previous high before getting back to it, in calendar days.",
                         "组合低于前高、直到重新回到前高所经历的最长时间（日历日）。"),
    "Value at risk (95%)": ("A bad day: on 95% of days the loss was smaller than this; on the worst 5% it was larger.",
                            "“糟糕的一天”：95% 的交易日亏损小于这个数，最差的 5% 交易日亏损更大。"),
    "Skew": ("Whether big moves tend to be gains (positive) or losses (negative). Negative skew means occasional sharp drops.",
             "大幅波动更多是上涨（正）还是下跌（负）。负偏度意味着偶尔会有急跌。"),
    "Kurtosis": ("How often extreme days happen compared with a normal distribution; above 0 means fatter tails, more surprises.",
                 "极端交易日出现的频率与正态分布相比；大于 0 表示尾部更厚，意外更多。"),
    "Sortino ratio": ("Like the Sharpe ratio, but only downside swings count as risk, so upside volatility is not penalized.",
                      "类似夏普比率，但只把下跌波动算作风险，上涨波动不扣分。"),
    "Calmar ratio": ("Annualized return divided by the max drawdown: how much was earned per unit of worst loss.",
                     "年化收益除以最大回撤：每承受一单位最大亏损换来多少收益。"),
    "Omega ratio": ("Total gains divided by total losses over all days. Above 1 means gains outweighed losses.",
                    "所有交易日的总收益除以总亏损。大于 1 表示收益多于亏损。"),
    "Tail ratio": ("Size of the best 5% of days compared with the worst 5%. Above 1 means big up days were larger than big down days.",
                   "最好的 5% 交易日与最差的 5% 交易日的幅度之比。大于 1 表示大涨日比大跌日幅度更大。"),
    "Common sense ratio": ("Profit factor times tail ratio; above 1 suggests the strategy's edge survives its bad days.",
                           "盈亏因子乘以尾部比率；大于 1 说明策略的优势能扛住糟糕的日子。"),
    "Rebalances with a gain": ("Share of holding periods (from one rebalance to the next) that ended with a profit.",
                               "持有期（从一次调仓到下一次）中以盈利结束的比例。"),
    "Months with a gain": ("Share of calendar months that ended with a profit.", "以盈利结束的自然月占比。"),
    "Annualised turnover": ("How much of the portfolio is bought and sold in a year, as a share of its value.",
                            "一年内买卖的金额占组合净值的比例。"),
    "Turnover per rebalance": ("Buys plus sells at a typical rebalance, as a share of the portfolio. Buying everything from cash is 100%, replacing the whole book about 200%.",
                               "一次调仓的买入加卖出占组合的比例。从现金全部买入是 100%，整个组合换一遍约 200%。"),
    "Total turnover": ("Buys plus sells over the whole window, as a share of the portfolio value.",
                       "整个区间内买入加卖出占组合净值的比例。"),
    "Traded notional": ("Total value of every trade over the window.", "整个区间内所有成交的总金额。"),
    "Fees paid": ("Fees and slippage the simulation charged, in money.", "模拟中扣除的手续费和滑点（金额）。"),
    "Max gross exposure": ("The largest total of long plus short positions held, as a share of the portfolio value.",
                           "持有的多头加空头总额占组合净值的最大比例。"),
    "Orders filled": ("Number of trades that actually happened.", "实际成交的笔数。"),
    "Round trips": ("Number of times a stock was bought and later fully sold (open ones included).",
                    "一只股票从买入到全部卖出算一次往返（含尚未平仓的）。"),
    "Round trips closed": ("Round trips that ended with the stock fully sold.", "已经全部卖出的往返次数。"),
    "Round trips open": ("Round trips still held at the last bar.", "最后一根 bar 时仍在持有的往返。"),
    "Open round-trip P&L": ("Unrealized profit of the positions still held.", "仍在持有的仓位的浮动盈亏。"),
    "Round-trip win rate": ("Share of closed round trips that made money.", "已平仓往返中盈利的比例。"),
    "Best round trip": ("Return of the best closed round trip.", "收益最好的一次已平仓往返。"),
    "Worst round trip": ("Return of the worst closed round trip.", "收益最差的一次已平仓往返。"),
    "Avg winning round trip": ("Average return of the winning round trips.", "盈利往返的平均收益。"),
    "Avg losing round trip": ("Average return of the losing round trips.", "亏损往返的平均收益。"),
    "Avg winning round-trip duration": ("How long winning positions were held on average.", "盈利仓位的平均持有时间。"),
    "Avg losing round-trip duration": ("How long losing positions were held on average.", "亏损仓位的平均持有时间。"),
    "Profit factor": ("Money made on winning round trips divided by money lost on losing ones. Above 1 is profitable.",
                      "盈利往返赚的钱除以亏损往返亏的钱。大于 1 表示整体盈利。"),
    "Expectancy": ("Average profit per closed round trip, in money.", "每次已平仓往返的平均盈利（金额）。"),
    "Orders rejected": ("Orders that could not be filled because the stock had no price at the next bar; the old holding was kept.",
                        "因下一根 bar 没有价格而无法成交的订单；原持仓保持不变。"),
    "Rebalances held after a failure": ("Rebalances where the portfolio optimizer failed, so the backtest kept the previous holdings.",
                                        "组合优化失败的调仓，回测沿用了之前的持仓。"),
    "Excess max drawdown": ("The worst stretch of falling behind the benchmark: the deepest fall of strategy / benchmark from its high.",
                            "落后基准最严重的一段：策略/基准 比值从高点下跌的最大幅度。"),
    "CAPM alpha": ("Return per year not explained by moving with the benchmark (the regression intercept).",
                   "无法用跟随基准波动来解释的年化收益（回归截距）。"),
    "Bars beating the benchmark": ("Share of days on which the portfolio did better than the benchmark.", "组合跑赢基准的交易日占比。"),
    "Rebalances beating the benchmark": ("Share of holding periods in which the portfolio did better than the benchmark.",
                                         "组合跑赢基准的持有期占比。"),
    "Months beating the benchmark": ("Share of calendar months in which the portfolio did better than the benchmark.",
                                     "组合跑赢基准的自然月占比。"),
    # factor attribution tiles
    "fa_total": ("The portfolio's growth per year in log terms; the factor attribution splits exactly this number into its sources.",
                 "组合每年的对数增长率；因子归因正是把这个数精确拆分到各个来源。"),
    "fa_factor": ("The part of the return explained by the portfolio's exposures to the risk model's factors: the market, industries and styles.",
                  "能用组合在风险模型因子上的暴露解释的收益：市场、行业和风格。"),
    "fa_rf_trading": ("Interest on the invested money (risk-free rate) plus everything trading changed: fills at the open, fees, slippage and idle cash.",
                      "投入资金的无风险利息，加上交易带来的一切变化：开盘成交、手续费、滑点和闲置现金。"),
    "fa_fvol": ("The volatility the risk model predicted for the portfolio held each day, averaged over the window and annualized.",
                "风险模型对每天所持组合预测的波动率，在区间内平均后年化。"),
    "fa_rvol": ("The volatility the portfolio actually had, annualized. Compare it with the forecast to judge the risk model.",
                "组合实际的年化波动率。与预测值对比可以判断风险模型准不准。"),
    "fa_cov": ("Share of the portfolio's holdings the risk model knows (has exposures for). The rest lands in Uncovered.",
               "风险模型能识别（有暴露数据）的持仓占比。其余部分计入“未覆盖”。"),
}

#: Chart explanations behind the ⓘ: key -> (English, Chinese).
CARD_TIPS = {
    "ret_part": ("Each bar is how much one source added to (blue) or took from (red) the yearly log growth. The bars add up to the grey Total. Country is the market, Industry and Style the portfolio's tilts, Specific the stock picks the factors cannot explain.",
                 "每根柱是一个来源对年化对数增长的贡献：蓝色为正、红色为负，所有柱加起来等于灰色的“合计”。国家（市场）是整体市场，行业和风格是组合的倾斜，特异（选股）是因子解释不了的个股收益。"),
    "risk_part": ("Each bar is how much one source adds to the predicted volatility (x-sigma-rho). They add up to the grey Forecast. A red bar means that source hedges the rest. The black bar is the volatility that actually happened.",
                  "每根柱是一个来源对预测波动率的贡献（x-sigma-rho），加起来等于灰色的“预测”。红色表示该来源在对冲其他部分。黑色柱是实际发生的波动率。"),
    "ret_time": ("The running total of each source over time. The black line is the portfolio itself (log value): the coloured lines always add up to it.",
                 "各来源随时间的累计贡献。黑线是组合本身（对数净值），彩色线在任何时刻加起来都等于它。"),
    "risk_time": ("The predicted volatility each month, split by source (the bars add up to the prediction), against the volatility that actually happened (dotted).",
                  "每个月的预测波动率按来源拆分（柱子加起来等于预测值），与实际发生的波动率（虚线）对比。"),
    "style_ret": ("Left: how strongly the portfolio leaned into each style (positive = more of it than the market). Right: what that lean earned or lost per year.",
                  "左：组合在每个风格上的倾斜程度（正 = 比市场更多）。右：这种倾斜每年带来的盈亏。"),
    "style_risk": ("How much each style added to the predicted volatility (left) and to the volatility that actually happened (right). Same order as the chart beside.",
                   "每个风格对预测波动率（左）和实际波动率（右）的贡献，顺序与左边的图相同。"),
    "ind_ret": ("The 10 industries that added the most and the 10 that cost the most per year. The number beside each name is the portfolio's average net weight in it.",
                "每年贡献最多的 10 个行业和拖累最多的 10 个行业。名称旁的数字是组合在该行业的平均净权重。"),
    "ind_risk": ("The 10 industries adding the most to the predicted volatility and the 10 reducing it most (hedging). The number is the average net weight.",
                 "对预测波动率贡献最大的 10 个行业和降低最多（起对冲作用）的 10 个行业。数字是平均净权重。"),
    "style_heat": ("Each row is a style, each column a week. Blue means the portfolio leaned into the style, red away from it; stronger colour, bigger lean.",
                   "每行是一个风格，每列是一周。蓝色表示组合偏向该风格，红色表示回避；颜色越深倾斜越大。"),
    "ret_risk": ("For each source, what it earned per year (dark) against how much of the actual volatility it caused (light). A source with a long light bar and a short dark one took risk without being paid.",
                 "每个来源每年赚到的收益（深色）与它造成的实际波动（浅色）的对比。浅色长、深色短，说明承担了风险却没有得到回报。"),
}

# ---------------------------------------------------------------------------
# charts, built once per language
# ---------------------------------------------------------------------------


def _style(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Inter, system-ui, -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif",
                  size=12, color=INK2),
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", x=0, y=1.0, xanchor="left", yanchor="bottom", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=MUTED))


def _new(height):
    fig = go.Figure()
    _style(fig, height)
    return fig


def _padded(values, pad=1.45):
    lo, hi = min(min(values), 0.0), max(max(values), 0.0)
    span = (hi - lo) or 1.0
    return [lo - span * (pad - 1) if lo < 0 else -span * 0.05, hi + span * (pad - 1) if hi > 0 else span * 0.05]


def _baseline(fig, axis="y"):
    """A solid line at 0, over the grid."""
    if axis == "y":
        fig.add_hline(y=0, line=dict(color=INK2, width=1.2), layer="above")
    else:
        fig.add_vline(x=0, line=dict(color=INK2, width=1.2), layer="above")


def _parts(attribution):
    group = attribution["group"].values
    log = attribution["factor_log_contribution"].values
    out = {k: log[:, group == k].sum(axis=1) for k, _ in PARTS[:3]}
    out.update({k: attribution["log_contribution"].sel(term=k).values for k, _ in PARTS[3:]})
    index = pd.DatetimeIndex(attribution["timestamp"].values)
    return {k: pd.Series(v, index=index) for k, v in out.items()}


def by_part(seg, lang):
    """Return by part: plain bars from the zero axis, and the total."""
    g, a = seg["group_annualized_log_return"], seg["annualized_log_return"]
    keys = [k for k, _ in PARTS]
    values = [g.get(k, 0.0) if i < 3 else a[k] for i, k in enumerate(keys)] + [a["total"]]
    labels = [w(k, lang) for k in keys] + [w("total", lang)]
    colours = [POS if v >= 0 else NEG for v in values[:-1]] + [TOTAL]
    fig = _new(330)
    fig.add_trace(go.Bar(x=labels, y=values, marker=dict(color=colours, cornerradius=3),
                         text=[f"{v:+.1%}" for v in values], textposition="outside", cliponaxis=False,
                         textfont=dict(color=INK2), name="",
                         hovertemplate="%{x}<br>%{y:+.2%} " + w("a_year", lang) + "<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=_padded(values, 1.2))
    fig.update_layout(showlegend=False, bargap=0.35)
    _baseline(fig)
    return fig


def risk_by_part(seg, lang):
    ante = seg["ex_ante_risk"]
    keys = [k for k, _ in PARTS[:4]]
    values = [ante["group_contribution"].get(k, 0.0) for k in keys[:3]] + [ante["contribution"]["specific"]]
    total, realized = ante["volatility"]["total"], seg["ex_post_risk"]["volatility"]
    labels = [w(k, lang) for k in keys] + [w("forecast", lang), w("realized", lang)]
    values = values + [total, realized]
    colours = ["#86b6ef" if v >= 0 else NEG for v in values[:4]] + [TOTAL, INK]
    fig = _new(330)
    fig.add_trace(go.Bar(x=labels, y=values, marker=dict(color=colours, cornerradius=3),
                         text=[f"{v:+.1%}" for v in values[:4]] + [f"{total:.1%}", f"{realized:.1%}"],
                         textposition="outside", cliponaxis=False, textfont=dict(color=INK2), name="",
                         hovertemplate="%{x}<br>%{y:.2%} " + w("ann_vol", lang) + "<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=_padded(values, 1.2))
    fig.update_layout(showlegend=False, bargap=0.35)
    _baseline(fig)
    return fig


def _dodge(values, gap):
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out.tolist()


def over_time(attribution, lang):
    parts = _parts(attribution)
    first, last = parts["country"].index[0], parts["country"].index[-1]
    fig = _new(380)
    labels = []
    for key, colour in PARTS:
        curve = parts[key].cumsum()
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=w(key, lang), mode="lines",
                                 line=dict(color=colour, width=2), hovertemplate=f"{w(key, lang)} %{{y:+.1%}}<extra></extra>"))
        labels.append((curve.values[-1], f"{w(key, lang)} {curve.values[-1]:+.0%}", INK2, colour))
    total = sum(parts.values()).cumsum()
    fig.add_trace(go.Scatter(x=total.index, y=total.values, name=w("log_nav", lang), mode="lines",
                             line=dict(color=INK, width=3), hovertemplate=f"{w('total', lang)} %{{y:+.1%}}<extra></extra>"))
    labels.append((total.values[-1], f"<b>{w('total', lang)} {total.values[-1]:+.0%}</b>", INK, INK))
    span = float(max(total.max(), max(p.cumsum().max() for p in parts.values())) -
                 min(total.min(), min(p.cumsum().min() for p in parts.values())))
    for (y, text, ink, colour), at in zip(labels, _dodge([y for y, *_ in labels], span * 0.055)):
        fig.add_annotation(x=last, y=at, text=f'<span style="color:{colour}">■</span> {text}', showarrow=False,
                           xanchor="left", xshift=8, font=dict(size=11, color=ink))
    fig.update_layout(hovermode="x unified", showlegend=False, margin=dict(l=10, r=150, t=10, b=10))
    fig.update_xaxes(range=[first, last])
    fig.update_yaxes(tickformat=".0%")
    _baseline(fig)
    return fig


def risk_over_time(attribution, lang):
    scale = np.sqrt(BARS_PER_YEAR)
    group = attribution["group"].values
    risk = attribution["factor_risk_contribution"].to_pandas()
    frame = pd.DataFrame({k: risk.loc[:, group == k].sum(axis=1, min_count=1) * scale for k, _ in PARTS[:3]})
    frame["specific"] = attribution["specific_risk_contribution"].to_pandas() * scale
    monthly = frame.resample("ME").mean()
    realized = attribution["return"].to_pandas().rolling(63).std() * scale
    fig = _new(380)
    for key, colour in PARTS[:4]:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[key], name=w(key, lang), marker=dict(color=colour),
                             hovertemplate=f"%{{x|%Y-%m}} {w(key, lang)} %{{y:.1%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=realized.index, y=realized.values, name=w("realized63", lang), mode="lines",
                             line=dict(color=INK, width=2, dash="dot"),
                             hovertemplate=f"%{{x|%Y-%m-%d}} {w('realized', lang)} %{{y:.1%}}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.1, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(tickformat=".0%", title=dict(text=w("ann_vol", lang), font=dict(size=11, color=MUTED)))
    _baseline(fig)
    return fig


def _styles(attribution, seg):
    names = [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == "style"]
    return sorted(names, key=lambda n: seg["factor_annualized_log_return"][n])


def _hbars(fig, labels, values, fmt, hover, row=None, col=None):
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h", marker=dict(color=[POS if v >= 0 else NEG for v in values], cornerradius=3),
                         text=[fmt(v) for v in values], textposition="outside", cliponaxis=False,
                         textfont=dict(color=INK2, size=11), hovertemplate=hover, name=""), **kw)
    fig.add_vline(x=0, line=dict(color=INK2, width=1.2), layer="above", **kw)


def style_return(seg, attribution, lang):
    names = _styles(attribution, seg)
    labels = [name_of(n, lang) for n in names]
    ex = [seg["style_mean_exposure"][n] for n in names]
    gr = [seg["factor_annualized_log_return"][n] for n in names]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=(w("mean_exp", lang), w("contrib", lang)))
    _style(fig, 380)
    _hbars(fig, labels, ex, lambda v: f"{v:+.2f}", "%{y}: %{x:.2f}<extra></extra>", 1, 1)
    _hbars(fig, labels, gr, lambda v: f"{v:+.1%}", "%{y}: %{x:+.2%} " + w("a_year", lang) + "<extra></extra>", 1, 2)
    fig.update_xaxes(range=_padded(ex), row=1, col=1)
    fig.update_xaxes(tickformat=".0%", range=_padded(gr), row=1, col=2)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def style_risk(seg, attribution, lang):
    names = _styles(attribution, seg)
    labels = [name_of(n, lang) for n in names]
    fc = [seg["ex_ante_risk"]["factor_contribution"][n] for n in names]
    rz = [seg["ex_post_risk"]["factor_contribution"][n] or 0.0 for n in names]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=(w("fc_risk", lang), w("rz_risk", lang)))
    _style(fig, 380)
    for col, values in ((1, fc), (2, rz)):
        _hbars(fig, labels, values, lambda v: f"{v:+.2%}", "%{y}: %{x:+.2%}<extra></extra>", 1, col)
        fig.update_xaxes(tickformat=".1%", range=_padded(values), row=1, col=col)
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10), bargap=0.3)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def industries(seg, attribution, lang, risk=False, n=10):
    names = [str(f) for f, g in zip(attribution["factor"].values, attribution["group"].values) if g == "industry"]
    source = seg["ex_ante_risk"]["factor_contribution"] if risk else seg["factor_annualized_log_return"]
    weight = attribution["exposure"].sel(factor=names).where(attribution["gross_weight"] > 0).mean("timestamp")
    weight = dict(zip(names, weight.values))
    ranked = sorted(names, key=lambda k: -source[k])
    shown = ranked[:n] + ranked[-n:]
    xs = [source[k] for k in shown]
    labels = [name_of(k, lang) for k in shown]
    fig = _new(460)
    unit = w("of_fc_vol", lang) if risk else w("a_year", lang)
    fig.add_trace(go.Bar(y=labels, x=xs, orientation="h", marker=dict(color=[POS if v >= 0 else NEG for v in xs], cornerradius=3),
                         text=[f"{v:+.2%}" for v in xs], textposition="outside", cliponaxis=False,
                         textfont=dict(size=10, color=INK2), customdata=[weight[k] for k in shown], name="",
                         hovertemplate="%{y}<br>%{x:+.2%} " + unit + "<br>" + w("mean_weight", lang) + " %{customdata:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickvals=labels,
                     ticktext=[f"{l}  <span style='color:{MUTED}'>{weight[k]:.0%}</span>" for l, k in zip(labels, shown)])
    fig.update_xaxes(tickformat=".2%" if risk else ".1%", nticks=6, range=_padded(xs, 1.3))
    fig.update_layout(showlegend=False, bargap=0.25)
    _baseline(fig, "x")
    return fig


def style_heatmap(seg, attribution, lang):
    names = sorted(_styles(attribution, seg), key=lambda n: -seg["factor_annualized_log_return"][n])
    exposure = attribution["exposure"].sel(factor=names).to_pandas()
    weekly = exposure[attribution["gross_weight"].to_pandas() > 0].resample("W-FRI").mean()
    lim = float(np.nanpercentile(np.abs(weekly.values), 98)) or 1.0
    fig = _new(360)
    fig.add_trace(go.Heatmap(z=weekly.T.values, x=weekly.index, y=[name_of(n, lang) for n in names], zmid=0,
                             zmin=-lim, zmax=lim, colorscale=[[0, NEG], [0.5, MID], [1, POS]], ygap=2,
                             colorbar=dict(title=dict(text=w("exposure", lang), font=dict(size=11, color=MUTED)), thickness=10),
                             hovertemplate="%{y} · " + w("week", lang) + " %{x|%Y-%m-%d}<br>" + w("exposure", lang) + " %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def return_vs_risk(seg, lang):
    g, a, post = seg["group_annualized_log_return"], seg["annualized_log_return"], seg["ex_post_risk"]
    labels, ret, risk = [], [], []
    for i, (key, _) in enumerate(PARTS):
        labels.append(w(key, lang))
        ret.append(g[key] if i < 3 else a[key])
        risk.append((post["group_contribution"][key] if i < 3 else post["term_contribution"][key]) or 0.0)
    fig = _new(340)
    for values, name, colour in ((ret, w("ret_bar", lang), POS), (risk, w("risk_bar", lang), "#86b6ef")):
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h", name=name, marker=dict(color=colour, cornerradius=3),
                             text=[f"{v:+.1%}" for v in values], textposition="outside", cliponaxis=False,
                             textfont=dict(size=10, color=INK2), hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=_padded(ret + risk, 1.15))
    _baseline(fig, "x")
    return fig


# ---------------------------------------------------------------------------
# the factor attribution page (F2), both languages
# ---------------------------------------------------------------------------

_counter = [0]


def _tx(en: str, zh: str) -> str:
    return f'<span data-en="{html.escape(en)}" data-zh="{html.escape(zh)}">{html.escape(en)}</span>'


def _tip(en: str, zh: str) -> str:
    return f'data-tip-en="{html.escape(en)}" data-tip-zh="{html.escape(zh)}"'


def _chart(build, *args) -> str:
    """The English figure as html, and both languages' specs for the switch."""
    _counter[0] += 1
    div_id = f"fa{_counter[0]}"
    en, zh = build(*args, "en"), build(*args, "zh")
    page = en.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                      config={"responsive": True, "displaylogo": False})
    specs = {"en": json.loads(en.to_json()), "zh": json.loads(zh.to_json())}
    return page + f'<script type="application/json" class="spec" data-for="{div_id}">{json.dumps(specs)}</script>'


def _card(title_en, title_zh, sub_en, sub_zh, tip_key, body):
    tip = CARD_TIPS[tip_key]
    return (f'<div class="card"><p class="q">{_tx(title_en, title_zh)} <span class="i" {_tip(*tip)}>ⓘ</span></p>'
            f'<p class="a">{_tx(sub_en, sub_zh)}</p>{body}</div>')


def _tiles(seg, scope_en, scope_zh):
    g, ea, ep, cov = seg["annualized_log_return"], seg["ex_ante_risk"], seg["ex_post_risk"], seg["coverage"]
    vol = ea["volatility"]
    share = vol["factor"] ** 2 / vol["total"] ** 2
    pct = lambda v: f"{100 * v:+.2f}%"

    def tile(key, en, zh, value, sub_en, sub_zh):
        return (f'<div class="tile" {_tip(*TIPS[key])}><div class="l">{_tx(en, zh)}</div><div class="v">{value}</div>'
                f'<div class="s">{_tx(sub_en, sub_zh)}</div></div>')

    return '<div class="tiles">' + "".join([
        tile("fa_total", "Log growth / yr", "年化对数增长", pct(g["total"]), scope_en, scope_zh),
        tile("fa_factor", "From factors", "来自因子", pct(g["factor"]), f"specific {pct(g['specific'])}", f"特异 {pct(g['specific'])}"),
        tile("fa_rf_trading", "Risk-free + trading", "无风险 + 交易", pct(g["risk_free"] + g["trading"]),
             f"uncovered {pct(g['uncovered'])}", f"未覆盖 {pct(g['uncovered'])}"),
        tile("fa_fvol", "Forecast vol", "预测波动率", f"{100 * vol['total']:.2f}%",
             f"{share:.0%} of variance from factors", f"{share:.0%} 的方差来自因子"),
        tile("fa_rvol", "Realized vol", "实现波动率", f"{100 * ep['volatility']:.2f}%", "ex post", "事后"),
        tile("fa_cov", "Coverage", "覆盖率", f"{100 * cov['mean_covered_weight']:.1f}%",
             f"min {cov['min_covered_weight']:.1%}", f"最低 {cov['min_covered_weight']:.1%}"),
    ]) + "</div>"


def f2(block, attribution, metrics) -> str:
    if br._has_in_sample(metrics):
        seg, en, zh = block["out_of_sample"], "out-of-sample", "样本外"
    else:
        seg, en, zh = block["whole"], "whole window", "全区间"
    pair = lambda a, b: f'<div class="row">{a}{b}</div>'
    return f"""<div class="fa">{pv_css()}{_tiles(seg, en, zh)}
<div class="row"><div class="colhead">{_tx("Return", "收益")}</div><div class="colhead">{_tx("Risk", "风险")}</div></div>
{pair(_card("Return by part", "收益按来源", f"Annualized log growth, {en}; the bars add up to the total.", f"年化对数增长，{zh}；各柱相加等于合计。", "ret_part", _chart(by_part, seg)),
      _card("Risk by part", "风险按来源", "Contribution to the forecast volatility (x-sigma-rho), and the realized volatility.", "对预测波动率的贡献（x-sigma-rho），以及实现波动率。", "risk_part", _chart(risk_by_part, seg)))}
{pair(_card("Return over time", "收益随时间", "Cumulative log contribution of each part; black: log NAV.", "各来源的累计对数贡献；黑线：对数净值。", "ret_time", _chart(over_time, attribution)),
      _card("Risk over time", "风险随时间", "Forecast volatility by part, monthly mean; dotted: realized 63-bar volatility.", "按来源拆分的预测波动率（月均）；虚线：63 根 bar 的实现波动率。", "risk_time", _chart(risk_over_time, attribution)))}
{pair(_card("Styles: exposure and return", "风格：暴露与收益", "Mean net exposure and annualized contribution.", "平均净暴露与年化贡献。", "style_ret", _chart(style_return, seg, attribution)),
      _card("Styles: risk", "风格：风险", "Contribution to forecast and to realized volatility, same order.", "对预测与实现波动率的贡献，顺序相同。", "style_risk", _chart(style_risk, seg, attribution)))}
{pair(_card("Industries: return", "行业：收益", "Best and worst 10 by annualized contribution; grey: mean net weight.", "年化贡献最高与最低的各 10 个；灰色：平均净权重。", "ind_ret", _chart(industries, seg, attribution)),
      _card("Industries: risk", "行业：风险", "Largest and smallest 10 contributions to forecast volatility.", "对预测波动率贡献最大与最小的各 10 个。", "ind_risk", _chart(lambda s, a, lang: industries(s, a, lang, risk=True), seg, attribution)))}
{_card("Style exposure over time", "风格暴露随时间", "Weekly mean net exposure; blue long, red short.", "每周平均净暴露；蓝为多，红为空。", "style_heat", _chart(style_heatmap, seg, attribution))}
{_card("Return against realized risk", "收益与实现风险", "Each part's annualized log growth and its contribution to realized volatility.", "各来源的年化对数增长及其对实现波动率的贡献。", "ret_risk", _chart(return_vs_risk, seg))}
</div>"""


def pv_css():
    return """<style>
  .fa { display:flex; flex-direction:column; gap:16px; }
  .fa .row { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; align-items:start; }
  .fa .card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px 16px; min-width:0; }
  .fa .q { font-size:13px; font-weight:600; color:#111827; margin:0 0 2px; }
  .fa .a { font-size:12px; color:#6b7280; margin:0 0 6px; }
  .fa .i { color:#9ca3af; cursor:help; font-weight:400; }
  .fa .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
  .fa .tile { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; cursor:help; }
  .fa .tile .l { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.05em; }
  .fa .tile .v { font-size:22px; font-weight:700; margin-top:2px; }
  .fa .tile .s { font-size:12px; color:#6b7280; }
  .fa .colhead { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
</style>"""


# ---------------------------------------------------------------------------
# page: language switch and hover explanations
# ---------------------------------------------------------------------------

_I18N_CSS = """<style>
  .lang { margin-left:auto; display:flex; background:#1e293b; border-radius:999px; padding:3px; }
  .lang button { border:0; background:none; color:#94a3b8; font:600 12px Inter,sans-serif; padding:4px 12px;
                 border-radius:999px; cursor:pointer; }
  .lang button.on { background:#fff; color:#0f172a; }
  #tip { position:fixed; z-index:1000; max-width:340px; background:#0f172a; color:#f8fafc; font:13px/1.5 Inter,
         'PingFang SC','Microsoft YaHei',sans-serif; padding:10px 12px; border-radius:8px; pointer-events:none;
         box-shadow:0 8px 24px rgba(0,0,0,.25); display:none; }
  #tip b { display:block; margin-bottom:4px; font-weight:600; }
  [data-tip-en] { cursor:help; }
  .kpi[data-tip-en]:hover, .tile[data-tip-en]:hover { border-color:#93c5fd; }
  table.metrics tbody th[data-tip-en] { text-decoration:underline dotted #9ca3af; text-underline-offset:3px; }
  html[lang=zh] body { font-family:Inter,'PingFang SC','Microsoft YaHei',sans-serif; }
</style>"""

_I18N_JS = """<script>
(function () {
  var ZH = __ZH__, RULES = __RULES__.map(function (r) { return [new RegExp(r[0]), r[1]]; });
  var CLEAN = {paper:'#fff', plot:'#fff', grid:'#eef0f3', zero:'#d1d5db', tick:'#6b7280', text:'#111827'};
  function zhOf(text) {
    var key = text.trim(); if (!key) return null;
    if (ZH[key]) return text.replace(key, ZH[key]);
    for (var i = 0; i < RULES.length; i++) if (RULES[i][0].test(key)) return text.replace(key, key.replace(RULES[i][0], RULES[i][1]));
    return null;
  }
  // Text nodes outside charts and outside the elements that carry both languages.
  var nodes = [];
  var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {acceptNode: function (n) {
    var p = n.parentElement;
    if (!p || p.closest('script,style,.plotly-graph-div,[data-en],#tip,.lang')) return NodeFilter.FILTER_REJECT;
    return zhOf(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
  }});
  while (walker.nextNode()) nodes.push({n: walker.currentNode, en: walker.currentNode.nodeValue, zh: zhOf(walker.currentNode.nodeValue)});
  // Charts the page wrote in both languages, and the others (axis titles, legends, buttons by dictionary).
  var specs = {};
  document.querySelectorAll('script.spec').forEach(function (s) { specs[s.dataset.for] = JSON.parse(s.textContent); });
  function chartLang(div, lang) {
    if (!window.Plotly) return;
    if (specs[div.id]) {
      var spec = specs[div.id][lang];
      Plotly.react(div, spec.data, spec.layout, {responsive: true, displaylogo: false}).then(function () { if (window.restyle) restyle(CLEAN); });
      return;
    }
    if (!div.__en) {
      var en = {names: (div.data || []).map(function (t) { return t.name; }), layout: {}};
      Object.keys(div.layout || {}).forEach(function (k) {
        if (/^[xy]axis\\d*$/.test(k) && div.layout[k].title && div.layout[k].title.text) en.layout[k + '.title.text'] = div.layout[k].title.text;
      });
      (div.layout.updatemenus || []).forEach(function (m, i) { (m.buttons || []).forEach(function (b, j) { en.layout['updatemenus[' + i + '].buttons[' + j + '].label'] = b.label; }); });
      div.__en = en;
    }
    var upd = {};
    Object.keys(div.__en.layout).forEach(function (k) { var v = div.__en.layout[k]; upd[k] = lang === 'zh' ? (zhOf(v) || ZH[v] || v) : v; });
    Plotly.relayout(div, upd);
    var names = div.__en.names.map(function (n) { return lang === 'zh' && n ? (zhOf(n) || n) : n; });
    if (names.length) Plotly.restyle(div, {name: names});
  }
  function setLang(lang) {
    document.documentElement.lang = lang;
    nodes.forEach(function (x) { x.n.nodeValue = lang === 'zh' ? x.zh : x.en; });
    document.querySelectorAll('[data-en]').forEach(function (e) { e.textContent = e.dataset[lang]; });
    document.querySelectorAll('.lang button').forEach(function (b) { b.classList.toggle('on', b.dataset.lang === lang); });
    document.querySelectorAll('.plotly-graph-div').forEach(function (d) { chartLang(d, lang); });
    try { localStorage.setItem('report-lang', lang); } catch (e) {}
    window.__lang = lang;
  }
  document.querySelectorAll('.lang button').forEach(function (b) { b.addEventListener('click', function () { setLang(b.dataset.lang); }); });
  // Hover explanations.
  var tip = document.getElementById('tip');
  document.addEventListener('mouseover', function (e) {
    var t = e.target.closest('[data-tip-en]'); if (!t) { tip.style.display = 'none'; return; }
    var lang = window.__lang || 'en', head = t.querySelector('.kl, .l');
    var title = head ? head.textContent : t.tagName === 'TH' ? t.textContent
      : t.classList.contains('i') ? t.parentElement.querySelector('[data-en]').textContent : '';
    tip.innerHTML = (title ? '<b>' + title + '</b>' : '') + t.getAttribute('data-tip-' + lang);
    tip.style.display = 'block';
  });
  document.addEventListener('mousemove', function (e) {
    if (tip.style.display !== 'block') return;
    var x = e.clientX + 14, y = e.clientY + 14, r = tip.getBoundingClientRect();
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });
  window.addEventListener('load', function () {
    var saved = 'en'; try { saved = localStorage.getItem('report-lang') || 'en'; } catch (e) {}
    var asked = new URLSearchParams(location.search).get('lang'); if (asked === 'zh' || asked === 'en') saved = asked;
    if (saved !== 'en') setTimeout(function () { setLang(saved); }, 50); else window.__lang = 'en';
  });
})();
</script>"""


def _tips_on_tables(page: str) -> str:
    """Give every metric row and KPI card its explanation in both languages."""
    def row(m):
        definition, label = html.unescape(m.group(1)), html.unescape(m.group(2))
        en, zh = TIPS.get(label, (definition, ZH.get(definition, definition)))
        return f'<th {_tip(en, zh)}>{m.group(2)}</th>'

    page = re.sub(r'<th title="([^"]*)">([^<]*)</th>', row, page)

    def card(m):
        label = html.unescape(m.group(1))
        en, zh = TIPS.get(label, ("", ""))
        return f'<div class="kpi" {_tip(en, zh)}><div class="kl">{m.group(1)}</div>' if en else m.group(0)

    return re.sub(r'<div class="kpi"><div class="kl">([^<]*)</div>', card, page)


def main(args_path: str, out_dir: str) -> None:
    kwargs = pickle.loads(Path(args_path).read_bytes())
    value = kwargs.pop("value")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = (br._document, br._factor_attribution_tables, br._factor_attribution_figure, br._figure_div)
    figure_div = br._figure_div

    def document(*a, **k):
        page = pv._b_document(*a, **k)
        page = re.sub(r'<div class="card"><!--FA-->(.*?)<!--/FA--></div>', r"\1", page, flags=re.S)
        page = _tips_on_tables(page)
        switch = '<div class="lang"><button data-lang="en" class="on">EN</button><button data-lang="zh">中文</button></div>'
        page = page.replace("</header>", switch + "</header>", 1)
        js = _I18N_JS.replace("__ZH__", json.dumps(ZH, ensure_ascii=False)).replace(
            "__RULES__", json.dumps([(a, re.sub(r"\\(\d)", r"$\1", b)) for a, b in ZH_RULES], ensure_ascii=False))
        page = page.replace("</head>", _I18N_CSS + "</head>", 1)
        opener = ("<script>window.addEventListener('load', function () { document.querySelectorAll('nav button')"
                  ".forEach(function (b) { if (b.textContent === 'Factor attribution') b.click(); }); });</script>")
        return page.replace("</body>", '<div id="tip"></div>' + js + opener + "</body>")

    br._document = document
    br._factor_attribution_tables = lambda block, attribution, metrics: "<!--FA-->" + f2(block, attribution, metrics) + "<!--/FA-->"
    br._factor_attribution_figure = lambda *a, **k: None
    br._figure_div = lambda fig: "" if fig is None else figure_div(fig)
    try:
        br.write_backtest_report(value, out / "report_f2_i18n.html", **kwargs)
    finally:
        br._document, br._factor_attribution_tables, br._factor_attribution_figure, br._figure_div = saved
    print(out / "report_f2_i18n.html")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
