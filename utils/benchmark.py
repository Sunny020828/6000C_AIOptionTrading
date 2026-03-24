import pandas as pd
import numpy as np
def load_and_prepare_data(filename='merged_df.csv'):
    # 读取 CSV 文件
    df = pd.read_csv(filename)

    # 转换日期格式
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['ExpDate'] = pd.to_datetime(df['ExpDate'], errors='coerce')

    # 删除缺失关键字段的行
    df = df.dropna(subset=['Date','ExpDate','StrikePrice','Type','dte','Close'])

    # 标准化期权类型
    df['Type'] = df['Type'].str.lower().replace({'c':'call','p':'put'})

    return df

df = load_and_prepare_data("merged_df.csv")
def generate_benchmark_signals(date, df, direction="long"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    day_data = df[df["Date"] == pd.to_datetime(date).normalize()]
    if day_data.empty:
        return []

    S = float(day_data["Close"].iloc[0])
    action = "BUY" if direction == "long" else "SELL"

    signals = [{
        "action": action,
        "contract": "BENCHMARK",
        "quantity": 1,
        "strategy": f"{direction}_benchmark",
        "leg": "benchmark_leg",
        "price_est": S,
    },
        {
            "meta": {
                "date": date,
                "scenario": f"{direction}_benchmark",
                "strategy_type": "benchmark",
                "underlying_price": S,
                "quantity": 1,
            }
        }]
    return signals


def calculate_benchmark_pnl(df, direction, start_date, end_date):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # 过滤区间
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    if df.empty:
        return None

    # 建仓价格（起始日收盘价）
    entry_price = df.loc[df["Date"] == pd.to_datetime(start_date), "Close"].iloc[0]
    sign = 1 if direction == "long" else -1

    # 计算每日累计PnL
    df["Total_PnL"] = sign * (df["Close"] - entry_price)
    df["Daily_PnL"] = df["Total_PnL"].diff().fillna(df["Total_PnL"])

    return df


import matplotlib.pyplot as plt


def plot_benchmark_pnl(pnl_df, direction):
    pnl_df["Month"] = pnl_df["Date"].dt.to_period("M")
    monthly_pnl = pnl_df.groupby("Month")["Daily_PnL"].sum()

    monthly_pnl.plot(kind="bar", figsize=(12, 6), title=f"{direction.capitalize()} Benchmark Monthly PnL")
    plt.ylabel("PnL")
    plt.xlabel("Month")
    plt.show()

    return monthly_pnl

def calculate_benchmark_pnl_and_sharpe(df, direction, start_date, end_date):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    # 过滤区间
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    if df.empty:
        return None, None, None

    # 建仓价格（起始日收盘价）
    entry_price = df.loc[df["Date"] == pd.to_datetime(start_date), "Close"].iloc[0]
    sign = 1 if direction == "long" else -1

    # 计算累计PnL
    df["Total_PnL"] = sign * (df["Close"] - entry_price)
    df["Daily_PnL"] = df["Total_PnL"].diff().fillna(df["Total_PnL"])

    # 日收益率（相对建仓价）
    df["Return"] = df["Daily_PnL"] / entry_price

    # 夏普比率
    mean_return = df["Return"].mean()
    std_return = df["Return"].std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else np.nan

    # 最终累计PnL
    final_pnl = df["Total_PnL"].iloc[-1]

    return df, final_pnl, sharpe_ratio


# Long benchmark
benchmark_long_df, final_long_pnl, sharpe_long = calculate_benchmark_pnl_and_sharpe(
    df, direction="long", start_date="2024-01-02", end_date="2026-03-02"
)
monthly_long = plot_benchmark_pnl(benchmark_long_df, direction="long")
print("Long Benchmark Final PnL:", final_long_pnl)
print("Long Benchmark Sharpe Ratio:", sharpe_long)

# Short benchmark
benchmark_short_df, final_short_pnl, sharpe_short = calculate_benchmark_pnl_and_sharpe(
    df, direction="short", start_date="2024-01-02", end_date="2026-03-02"
)
monthly_short = plot_benchmark_pnl(benchmark_short_df, direction="short")
print("Short Benchmark Final PnL:", final_short_pnl)
print("Short Benchmark Sharpe Ratio:", sharpe_short)

