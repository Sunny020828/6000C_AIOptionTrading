import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


# ====================== Data Loading & Preprocessing ======================

def load_and_prepare_data(filename='merged_df.csv'):
    """
    Load CSV file, clean column names, parse dates, return processed DataFrame
    """
    df = pd.read_csv(filename)

    # Keep original column names but ensure consistency
    # Expected columns: Series, Date, Type, StrikePrice, ExpDate, SettlementPrice,
    # ImpliedVolatility, dte, Close, High, Low, Open, Volume, Returns, F,
    # Open_vhsi, High_vhsi, Low_vhsi, Close_vhsi

    # Rename columns to match code expectations (lowercase with underscores)
    column_mapping = {
        'Series': 'series',
        'Date': 'date',
        'Type': 'type',
        'StrikePrice': 'strikeprice',
        'ExpDate': 'expdate',
        'SettlementPrice': 'settlementprice',
        'ImpliedVolatility': 'impliedvolatility',
        'dte': 'dte',
        'Close': 'close',
        'High': 'high',
        'Low': 'low',
        'Open': 'open',
        'Volume': 'volume',
        'Returns': 'returns',
        'F': 'f',
        'Open_vhsi': 'open_vhsi',
        'High_vhsi': 'high_vhsi',
        'Low_vhsi': 'low_vhsi',
        'Close_vhsi': 'close_vhsi'
    }

    # Only rename columns that exist
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert date columns to datetime type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['expdate'] = pd.to_datetime(df['expdate'], errors='coerce')

    # Delete rows with failed date conversion
    df = df.dropna(subset=['date', 'expdate'])

    # Ensure numeric columns are float type
    numeric_cols = ['strikeprice', 'settlementprice', 'impliedvolatility', 'dte', 'close',
                    'volume', 'open', 'high', 'low', 'returns', 'f', 'open_vhsi',
                    'high_vhsi', 'low_vhsi', 'close_vhsi']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Delete rows with empty key fields
    df = df.dropna(subset=['strikeprice', 'type', 'expdate', 'dte', 'close'])

    # Convert option type to lowercase and standardize
    df['type'] = df['type'].str.lower()
    df = df[df['type'].isin(['call', 'put', 'c', 'p'])]

    # Standardize type values
    df['type'] = df['type'].replace({'c': 'call', 'p': 'put'})

    return df


# ====================== Helper Functions ======================

def get_options_for_date(df, date):
    """Get all option data for a specific date"""
    return df[df['date'] == date].copy()


def get_target_expiry(df, date, min_dte=30, max_dte=45):
    """
    Select contract month closest to 30-45 days to expiration
    If no conditions are met, take the smallest dte >= min_dte
    """
    day_data = get_options_for_date(df, date)
    expiries = day_data[['expdate', 'dte']].drop_duplicates()
    candidates = expiries[(expiries['dte'] >= min_dte) & (expiries['dte'] <= max_dte)]
    if not candidates.empty:
        # Select expiration near the middle value
        target = candidates.iloc[(candidates['dte'] - (min_dte + max_dte) / 2).abs().argmin()]['expdate']
    else:
        # Relax condition: take nearest expiration with dte >= min_dte
        candidates = expiries[expiries['dte'] >= min_dte]
        if not candidates.empty:
            target = candidates.nsmallest(1, 'dte')['expdate'].iloc[0]
        else:
            target = None
    return target


def find_closest_strike(strikes, target):
    """Return the strike price closest to the target value"""
    return min(strikes, key=lambda x: abs(x - target))


def black_delta(S, K, T, r, sigma, option_type):
    """
    Calculate Delta for European options using Black-76 model (futures option approximation)
    S: Underlying asset price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (simplified to 0)
    sigma: Implied volatility
    option_type: 'call' or 'put'
    """
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan
    d1 = (np.log(S / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    return delta


def estimate_delta(row, S, r=0.02):
    """Estimate Delta based on option data row (return NaN if no implied volatility)"""
    T = row['dte'] / 365.0
    sigma = row.get('impliedvolatility', np.nan)
    if pd.isna(sigma) or sigma <= 0:
        return np.nan
    return black_delta(S, row['strikeprice'], T, r, sigma, row['type'])


def filter_liquid_options(df, min_volume=10, min_oi=10):
    """Liquidity filtering: volume and open interest thresholds"""
    if 'volume' in df.columns:
        df = df[df['volume'] >= min_volume]
    if 'open' in df.columns:  # open represents open interest
        df = df[df['open'] >= min_oi]
    return df


def calculate_position_size(max_loss_per_trade, account_value=1_000_000, risk_pct=0.01):
    """
    Calculate number of tradable contracts based on maximum loss per contract
    Returns number of contracts (integer, rounded down)
    """
    risk_amount = account_value * risk_pct
    if max_loss_per_trade <= 0:
        return 0
    qty = int(risk_amount // max_loss_per_trade)
    return max(qty, 1)


# ====================== Spread Generation Functions (Single Spread) ======================

def generate_single_spread(scenario_type, opt_type, delta_buy_target, delta_sell_target,
                           S, subset, use_pct, contract_multiplier, account_value, risk_pct):
    """
    Generate single spread contract selection and signals based on Delta targets
    Returns (buy_contract, sell_contract, quantity, max_loss_per_spread) or None
    """
    # Calculate Delta
    if use_pct:
        # Use percentage approximation (simplified, can be customized)
        if opt_type == 'call':
            # Buy leg Delta target corresponds to strike slightly above S, sell leg higher
            buy_pct_map = {0.32: 0.02, 0.42: 0.01}  # Example mapping
            sell_pct_map = {0.17: 0.05, 0.22: 0.03}
            buy_pct = 0.02 if delta_buy_target > 0.3 else 0.03
            sell_pct = 0.05 if delta_sell_target > 0.15 else 0.06
            buy_target = S * (1 + buy_pct)
            sell_target = S * (1 + sell_pct)
        else:  # put
            buy_pct = 0.02 if abs(delta_buy_target) > 0.3 else 0.03
            sell_pct = 0.05 if abs(delta_sell_target) > 0.15 else 0.06
            buy_target = S * (1 - buy_pct)
            sell_target = S * (1 - sell_pct)

        strikes = sorted(subset['strikeprice'].unique())
        k_buy = find_closest_strike(strikes, buy_target)
        k_sell = find_closest_strike(strikes, sell_target)
    else:
        # Select strikes based on Delta
        subset['Delta_abs'] = subset['Delta'].abs()

        # Buy leg (further out-of-the-money, smaller absolute Delta)
        buy_candidates = subset[subset['Delta_abs'] <= abs(delta_buy_target) * 1.2]
        if buy_candidates.empty:
            buy_candidates = subset.nsmallest(1, 'Delta_abs')
        k_buy = buy_candidates.loc[(buy_candidates['Delta_abs'] - abs(delta_buy_target)).abs().idxmin(), 'strikeprice']

        # Sell leg (closer to at-the-money, larger absolute Delta)
        sell_candidates = subset[subset['Delta_abs'] >= abs(delta_sell_target) * 0.8]
        if sell_candidates.empty:
            sell_candidates = subset.nlargest(1, 'Delta_abs')
        k_sell = sell_candidates.loc[
            (sell_candidates['Delta_abs'] - abs(delta_sell_target)).abs().idxmin(), 'strikeprice']

    # Get contracts
    buy_contract = subset[subset['strikeprice'] == k_buy].iloc[0]
    sell_contract = subset[subset['strikeprice'] == k_sell].iloc[0]

    # Ensure two contracts are different
    if buy_contract['strikeprice'] == sell_contract['strikeprice']:
        strikes = sorted(subset['strikeprice'].unique())
        idx = strikes.index(k_buy)
        if idx > 0:
            k_buy = strikes[idx - 1] if opt_type == 'call' and k_buy > k_sell else strikes[
                min(idx + 1, len(strikes) - 1)]
        else:
            k_buy = strikes[1]
        buy_contract = subset[subset['strikeprice'] == k_buy].iloc[0]

    # Liquidity filtering
    buy_df = filter_liquid_options(pd.DataFrame([buy_contract]))
    sell_df = filter_liquid_options(pd.DataFrame([sell_contract]))
    if buy_df.empty or sell_df.empty:
        return None
    buy_contract = buy_df.iloc[0]
    sell_contract = sell_df.iloc[0]

    # Get price - using SettlementPrice
    price_col = 'settlementprice'
    if price_col not in buy_contract.index:
        price_col = 'close' if 'close' in buy_contract.index else None
    if price_col is None:
        return None

    buy_price = buy_contract[price_col]
    sell_price = sell_contract[price_col]

    # Calculate maximum loss
    if 'credit' in scenario_type:
        spread_width = abs(k_sell - k_buy)
        net_credit = sell_price - buy_price
        max_loss_per_spread = spread_width - net_credit
    else:  # debit spread
        net_debit = buy_price - sell_price
        max_loss_per_spread = net_debit

    max_loss_per_contract = max_loss_per_spread * contract_multiplier
    quantity = calculate_position_size(max_loss_per_contract, account_value, risk_pct)
    if quantity == 0:
        return None

    return {
        'buy_contract': buy_contract,
        'sell_contract': sell_contract,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'k_buy': k_buy,
        'k_sell': k_sell,
        'quantity': quantity,
        'max_loss_per_spread': max_loss_per_spread,
        'scenario_type': scenario_type
    }


# ====================== Main Strategy Function ======================

def generate_trade_signals(scenario, date, df, account_value=1_000_000, risk_pct=0.01):
    """
    Generate trading signals based on scenario and date
    Returns list of signals (each signal as dictionary)
    """
    # 1. Get data for the day
    day_data = get_options_for_date(df, date)
    if day_data.empty:
        print(f"[{date}] No data")
        return []

    S = day_data['close'].iloc[0]

    # 2. Select target expiration
    target_exp = get_target_expiry(day_data, date)
    if target_exp is None:
        print(f"[{date}] No suitable expiration")
        return []

    # 3. Determine strategy parameters based on scenario
    # Scenario configuration dictionary: strategy type, main option type
    scenario_config = {
        'bull_highvol': {'type': 'put_credit_spread', 'opt': 'put', 'buy_delta': 0.12, 'sell_delta': 0.28},
        'bull_lowvol': {'type': 'call_debit_spread', 'opt': 'call', 'buy_delta': 0.32, 'sell_delta': 0.17},
        'bull_medianvol': {'type': 'call_debit_spread', 'opt': 'call', 'buy_delta': 0.42, 'sell_delta': 0.22},
        'bear_highvol': {'type': 'call_credit_spread', 'opt': 'call', 'buy_delta': 0.12, 'sell_delta': 0.28},
        'bear_lowvol': {'type': 'put_debit_spread', 'opt': 'put', 'buy_delta': -0.32, 'sell_delta': -0.17},
        'bear_medianvol': {'type': 'put_debit_spread', 'opt': 'put', 'buy_delta': -0.42, 'sell_delta': -0.22},
        'range_highvol': {'type': 'iron_condor', 'opt': None,
                          'put_buy_delta': 0.10, 'put_sell_delta': 0.22,
                          'call_buy_delta': 0.10, 'call_sell_delta': 0.22},
        'range_lowvol': {'type': 'iron_condor',
                         'put_buy_delta': 0.05, 'put_sell_delta': 0.15,
                         'call_buy_delta': 0.05, 'call_sell_delta': 0.15},
        'range_medianvol': {'type': 'iron_condor',
                            'put_buy_delta': 0.10, 'put_sell_delta': 0.25,
                            'call_buy_delta': 0.10, 'call_sell_delta': 0.25}
    }

    if scenario not in scenario_config:
        print(f"[{date}] Unknown scenario: {scenario}")
        return []

    config = scenario_config[scenario]
    contract_multiplier = 50  # Hang Seng Index option multiplier, adjust as needed

    # 4. If iron_condor (range scenario), handle put and call sides separately
    if config['type'] == 'iron_condor':
        # Get put side data
        put_subset = day_data[(day_data['expdate'] == target_exp) & (day_data['type'] == 'put')].copy()
        call_subset = day_data[(day_data['expdate'] == target_exp) & (day_data['type'] == 'call')].copy()

        if put_subset.empty or call_subset.empty:
            print(f"[{date}] Iron condor missing put or call options")
            return []

        # Calculate Delta (if possible)
        for subset in [put_subset, call_subset]:
            subset['Delta'] = subset.apply(lambda row: estimate_delta(row, S), axis=1)

        # Determine whether to use percentage approximation
        use_pct_put = put_subset['Delta'].isna().all()
        use_pct_call = call_subset['Delta'].isna().all()

        # Generate put side spread (put credit spread)
        put_result = generate_single_spread(
            scenario_type='put_credit_spread',
            opt_type='put',
            delta_buy_target=config['put_buy_delta'],
            delta_sell_target=config['put_sell_delta'],
            S=S, subset=put_subset, use_pct=use_pct_put,
            contract_multiplier=contract_multiplier,
            account_value=account_value, risk_pct=risk_pct
        )

        # Generate call side spread (call credit spread)
        call_result = generate_single_spread(
            scenario_type='call_credit_spread',
            opt_type='call',
            delta_buy_target=config['call_buy_delta'],
            delta_sell_target=config['call_sell_delta'],
            S=S, subset=call_subset, use_pct=use_pct_call,
            contract_multiplier=contract_multiplier,
            account_value=account_value, risk_pct=risk_pct
        )

        if put_result is None or call_result is None:
            print(f"[{date}] Iron condor contract selection failed")
            return []

        # Use the smaller quantity to make both sides consistent
        qty = min(put_result['quantity'], call_result['quantity'])
        if qty == 0:
            return []

        # Build signals
        signals = []
        # Put side
        signals.append({
            'action': 'SELL',
            'contract': put_result['sell_contract']['series'],
            'quantity': qty,
            'strategy': scenario,
            'leg': 'put_sell',
            'price_est': put_result['sell_price']
        })
        signals.append({
            'action': 'BUY',
            'contract': put_result['buy_contract']['series'],
            'quantity': qty,
            'strategy': scenario,
            'leg': 'put_buy',
            'price_est': put_result['buy_price']
        })
        # Call side
        signals.append({
            'action': 'SELL',
            'contract': call_result['sell_contract']['series'],
            'quantity': qty,
            'strategy': scenario,
            'leg': 'call_sell',
            'price_est': call_result['sell_price']
        })
        signals.append({
            'action': 'BUY',
            'contract': call_result['buy_contract']['series'],
            'quantity': qty,
            'strategy': scenario,
            'leg': 'call_buy',
            'price_est': call_result['buy_price']
        })

        # Metadata
        signals.append({
            'meta': {
                'date': date,
                'scenario': scenario,
                'strategy_type': 'iron_condor',
                'underlying_price': S,
                'expiry': target_exp,
                'put_sell_strike': put_result['k_sell'],
                'put_buy_strike': put_result['k_buy'],
                'call_sell_strike': call_result['k_sell'],
                'call_buy_strike': call_result['k_buy'],
                'quantity': qty,
                'put_max_loss': put_result['max_loss_per_spread'],
                'call_max_loss': call_result['max_loss_per_spread']
            }
        })

        return signals

    else:
        # Single spread scenario (bull/bear)
        opt_type = config['opt']
        subset = day_data[(day_data['expdate'] == target_exp) & (day_data['type'] == opt_type)].copy()
        if subset.empty:
            print(f"[{date}] No {opt_type} options")
            return []

        subset['Delta'] = subset.apply(lambda row: estimate_delta(row, S), axis=1)
        use_pct = subset['Delta'].isna().all()

        result = generate_single_spread(
            scenario_type=config['type'],
            opt_type=opt_type,
            delta_buy_target=config['buy_delta'],
            delta_sell_target=config['sell_delta'],
            S=S, subset=subset, use_pct=use_pct,
            contract_multiplier=contract_multiplier,
            account_value=account_value, risk_pct=risk_pct
        )

        if result is None:
            print(f"[{date}] Spread selection failed")
            return []

        qty = result['quantity']
        signals = []
        if 'credit' in config['type']:
            # Credit spread: sell (main income), buy (protection)
            signals.append({
                'action': 'SELL',
                'contract': result['sell_contract']['series'],
                'quantity': qty,
                'strategy': scenario,
                'leg': 'sell_leg',
                'price_est': result['sell_price']
            })
            signals.append({
                'action': 'BUY',
                'contract': result['buy_contract']['series'],
                'quantity': qty,
                'strategy': scenario,
                'leg': 'buy_leg',
                'price_est': result['buy_price']
            })
        else:
            # Debit spread: buy (main direction), sell (reduce cost)
            signals.append({
                'action': 'BUY',
                'contract': result['buy_contract']['series'],
                'quantity': qty,
                'strategy': scenario,
                'leg': 'buy_leg',
                'price_est': result['buy_price']
            })
            signals.append({
                'action': 'SELL',
                'contract': result['sell_contract']['series'],
                'quantity': qty,
                'strategy': scenario,
                'leg': 'sell_leg',
                'price_est': result['sell_price']
            })

        signals.append({
            'meta': {
                'date': date,
                'scenario': scenario,
                'strategy_type': config['type'],
                'underlying_price': S,
                'expiry': target_exp,
                'buy_strike': result['k_buy'],
                'sell_strike': result['k_sell'],
                'quantity': qty,
                'max_loss_per_spread': result['max_loss_per_spread']
            }
        })

        return signals


# ====================== Main Program Example ======================

if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data('merged_df.csv')
    print("Data loading complete, total rows:", len(df))
    print("Date range:", df['date'].min(), "to", df['date'].max())
    print("\nColumns in processed dataframe:", df.columns.tolist())

    # Check available dates
    available_dates = df['date'].dt.date.unique()
    print("\nAvailable dates:", sorted(available_dates)[:10], "...")  # Show first 10 dates

    # Select a test date (make sure it exists in your data)
    if len(available_dates) > 0:
        test_date = pd.to_datetime(available_dates[0])  # Use first available date
        print(f"\nUsing test date: {test_date}")
    else:
        print("No dates available in data")
        exit()

    # Nine scenarios
    scenarios = [
        'bull_highvol', 'bull_lowvol', 'bull_medianvol',
        'bear_highvol', 'bear_lowvol', 'bear_medianvol',
        'range_highvol', 'range_lowvol', 'range_medianvol'
    ]

    for sc in scenarios:
        print(f"\n===== Scenario: {sc} =====")
        signals = generate_trade_signals(sc, test_date, df, account_value=1_000_000, risk_pct=0.01)
        for sig in signals:
            if 'meta' in sig:
                print("Strategy metadata:", sig['meta'])
            else:
                print(
                    f"Action: {sig['action']}, Contract: {sig['contract']}, Quantity: {sig['quantity']}, Estimated price: {sig['price_est']:.2f}")