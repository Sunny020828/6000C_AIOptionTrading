def generate_benchmark_signals(date, df, benchmark_type="long_atm_call"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["ExpDate"] = pd.to_datetime(df["ExpDate"], errors="coerce").dt.normalize()
    date = pd.to_datetime(date).normalize()

    # 找最近的交易日
    day_data, actual_date = get_options_for_date(df, date, return_actual_date=True, max_shift_days=None)
    if day_data.empty:
        return []

    # 标的价格
    S = float(day_data["Close"].iloc[0])
    strikes = day_data["StrikePrice"].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - S))

    # 选择到期日更远的合约（至少 30 天）
    expiries = day_data[["ExpDate", "dte"]].drop_duplicates()
    candidates = expiries[expiries["dte"] >= 30]
    if candidates.empty:
        target_exp = expiries["ExpDate"].max()
    else:
        target_exp = candidates["ExpDate"].min()

    # 根据类型选择合约
    if benchmark_type == "long_atm_call":
        subset = day_data[(day_data["Type"] == "call") & (day_data["ExpDate"] == target_exp)]
    else:
        subset = day_data[(day_data["Type"] == "put") & (day_data["ExpDate"] == target_exp)]

    if subset.empty:
        return []

    # 找最接近标的价格的合约
    chosen = subset.iloc[(subset["StrikePrice"] - S).abs().argmin()]

    # 保证 contract 名字和数据一致
    contract = str(chosen["Series"]).strip().upper()

    # 如果 SettlementPrice 缺失，用 Close 替代
    price_est = chosen["SettlementPrice"] if "SettlementPrice" in chosen and pd.notna(chosen["SettlementPrice"]) else chosen["Close"]

    return [{
        "action": "BUY" if benchmark_type == "long_atm_call" else "SELL",
        "contract": contract,
        "quantity": 1,
        "strategy": "benchmark_long_call" if benchmark_type == "long_atm_call" else "benchmark_short_put",
        "price_est": price_est,
        "meta": {"date": actual_date, "type": benchmark_type}
    }]


import matplotlib.pyplot as plt

def compare_long_short(option_df, trade_date, start_date, end_date):
    # Long ATM Call
    long_signals = generate_benchmark_signals(trade_date, option_df, benchmark_type="long_atm_call")
    long_pnl = mark_signals_to_market(option_df, long_signals, start_date, end_date)
    if not long_pnl.empty and "Total_PnL" in long_pnl.columns:
        long_pnl["Long_ATM_Call_PnL"] = long_pnl["Total_PnL"]

    # Short ATM Put
    short_signals = generate_benchmark_signals(trade_date, option_df, benchmark_type="short_atm_put")
    short_pnl = mark_signals_to_market(option_df, short_signals, start_date, end_date)
    if not short_pnl.empty and "Total_PnL" in short_pnl.columns:
        short_pnl["Short_ATM_PnL"] = short_pnl["Total_PnL"]

    return long_pnl, short_pnl

def backtest_long_short(option_df, start_date, end_date, freq="ME", hold_months=1):
    all_results = []
    trade_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    for trade_date in trade_dates:
        # 用 get_options_for_date 找最近的交易日
        day_data, actual_date = get_options_for_date(
            option_df,
            trade_date,
            return_actual_date=True,
            max_shift_days=None,
            forward_only=False
        )

        if day_data.empty:
            print(f"{trade_date} 没有数据，跳过")
            continue

        print(f"\n建仓目标日期: {trade_date}, 实际交易日: {actual_date}, day_data 行数: {len(day_data)}")

        # 生成信号
        long_signals = generate_benchmark_signals(actual_date, option_df, benchmark_type="long_atm_call")
        short_signals = generate_benchmark_signals(actual_date, option_df, benchmark_type="short_atm_put")

        print(f"生成 Long 信号数: {len(long_signals)}, Short 信号数: {len(short_signals)}")

        if long_signals:
            print("Long 信号详情:", long_signals)
        if short_signals:
            print("Short 信号详情:", short_signals)

        # 计算 PnL
        long_pnl = mark_signals_to_market(option_df, long_signals, actual_date, actual_date + pd.DateOffset(months=hold_months))
        short_pnl = mark_signals_to_market(option_df, short_signals, actual_date, actual_date + pd.DateOffset(months=hold_months))

        print(f"{actual_date} Long PnL 行数: {len(long_pnl)}, Short PnL 行数: {len(short_pnl)}")

        if not long_pnl.empty:
            long_pnl["TradeDate"] = actual_date
            all_results.append(long_pnl)

        if not short_pnl.empty:
            short_pnl["TradeDate"] = actual_date
            all_results.append(short_pnl)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def summarize_monthly_pnl(results):
    if results.empty:
        return pd.DataFrame()

    # 按月份汇总
    results["Month"] = results["Date"].dt.to_period("M")
    monthly_pnl = results.groupby(["Month"]).last()  # 每月最后一天的累计 PnL
    monthly_pnl = monthly_pnl[["Long_ATM_Call_PnL", "Short_ATM_PnL"]].fillna(0)

    # 计算每月收益（环比差值）
    monthly_pnl = monthly_pnl.diff().fillna(monthly_pnl)

    return monthly_pnl

def plot_monthly_pnl(monthly_pnl):
    if monthly_pnl.empty:
        print("没有有效的月度 PnL 数据")
        return

    plt.figure(figsize=(10,6))
    plt.plot(monthly_pnl.index.to_timestamp(), monthly_pnl["Long_ATM_Call_PnL"], label="Long ATM Call", linewidth=2)
    plt.plot(monthly_pnl.index.to_timestamp(), monthly_pnl["Short_ATM_PnL"], label="Short ATM Put", linewidth=2, linestyle="--")
    plt.title("每月 PnL 对比")
    plt.xlabel("月份")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)
    plt.show()
