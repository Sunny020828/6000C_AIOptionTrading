def generate_benchmark_signals(date, df, benchmark_type="long_atm_call"):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    date = pd.to_datetime(date).normalize()

    day_data = df[df["Date"] == date]
    if day_data.empty:
        return []

    # 找到平值期权（最接近标的价格的行权价）
    S = float(day_data["Close"].iloc[0])
    strikes = day_data["StrikePrice"].unique()
    atm_strike = min(strikes, key=lambda x: abs(x - S))

    if benchmark_type == "long_atm_call":
        atm_call = day_data[(day_data["StrikePrice"] == atm_strike) & (day_data["Type"] == "call")].iloc[0]
        return [{
            "action": "BUY",
            "contract": atm_call["Series"],
            "quantity": 1,
            "strategy": "benchmark_long_call",
            "price_est": atm_call["SettlementPrice"],
            "meta": {"date": date, "type": "long_atm_call"}
        }]
    elif benchmark_type == "short_atm_put":
        atm_put = day_data[(day_data["StrikePrice"] == atm_strike) & (day_data["Type"] == "put")].iloc[0]
        return [{
            "action": "SELL",
            "contract": atm_put["Series"],
            "quantity": 1,
            "strategy": "benchmark_short_put",
            "price_est": atm_put["SettlementPrice"],
            "meta": {"date": date, "type": "short_atm_put"}
        }]
