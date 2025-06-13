from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
import multiprocessing as mp 
import time

def basket_key(basket):
    return tuple(sorted(basket))

def get_coint_baskets(
    columns: list,
    data: pd.DataFrame | np.ndarray,
    significance_level: float,
    min_basket_size: int = 2,
    verbose: bool = True,
    seen_baskets: set = None,
    cointegrated_baskets: dict = None
):
    """
    Recursively finds cointegrated baskets by trimming assets with small eigenvector values.
    Returns (seen_baskets, cointegrated_baskets).
    cointegrated_baskets is a dict mapping basket_key -> list of eigenvectors (np.ndarray).
    """
    # Helper to create a canonical basket key
    def basket_key(basket):
        return tuple(sorted(basket))

    # Initialize sets/dicts if needed
    if seen_baskets is None:
        seen_baskets = set()
    if cointegrated_baskets is None:
        cointegrated_baskets = dict()

    current_basket = basket_key(tuple(columns))

    # If we've already seen this basket, return immediately
    if current_basket in seen_baskets:
        return seen_baskets, cointegrated_baskets

    # Mark this basket as seen
    seen_baskets.add(current_basket)

    # If basket is too small, stop recursion
    if len(columns) < min_basket_size:
        return seen_baskets, cointegrated_baskets

    # Prepare data and Johansen test
    log_data = np.log(np.array(data[list(columns)].values))
    sig_idx = {0.01: 0, 0.05: 1, 0.10: 2}[significance_level]

    try:
        result = coint_johansen(log_data, det_order=0, k_ar_diff=1)
    except Exception as e:
        if verbose:
            print(f"Johansen test failed for basket {current_basket}: {e}")
        return seen_baskets, cointegrated_baskets

    # Find cointegration rank
    rank = 0
    for i, stat in enumerate(result.lr1):
        if stat > result.cvt[i, sig_idx]:
            rank = i + 1
    if verbose:
        print(f"Rank from test for basket {current_basket}: {rank}")

    # If no cointegration, stop recursion
    if rank == 0:
        return seen_baskets, cointegrated_baskets

    # For each accepted eigenvector, try to trim and recurse
    trimmed = False
    for i in range(rank):
        eigvec = result.evec[:, i]
        max_val = np.max(np.abs(eigvec))
        trim_indices = np.where(np.abs(eigvec) < 0.05 * max_val)[0]
        trimmed_columns = [col for idx, col in enumerate(columns) if idx not in trim_indices]
        if len(trim_indices) > 0 and len(trimmed_columns) >= min_basket_size:
            # Recurse on trimmed basket
            seen_baskets, cointegrated_baskets = get_coint_baskets(
                trimmed_columns, data, significance_level, min_basket_size,
                verbose, seen_baskets, cointegrated_baskets
            )
            trimmed = True

    # If no further trimming was possible, add this basket as cointegrated
    if not trimmed:
        # Save all eigenvectors up to the cointegration rank for this basket
        eigvecs = [result.evec[:, i].copy() for i in range(rank)]
        cointegrated_baskets[current_basket] = eigvecs
        if verbose:
            print(f"Added cointegrated basket: {current_basket} with {rank} eigenvector(s)")

    return seen_baskets, cointegrated_baskets

def vectorized_cointegrated_basket_backtest(
    data: pd.DataFrame,
    basket: tuple[str, ...],
    beta_refresh_freq: int,
    spread_window: int,
    cash_start: float,
    notional: float,
    trade_freq: int,
    execution_delay: int,
    enter_zscore: float = 2.0,
    exit_zscore: float = 0.3
):
    """
    Vectorized backtest for cointegrated basket trading.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataframe containing all price columns.
    basket : tuple of str
        The tuple of column names to use for trading (e.g. ('bidPrice0_random_normal', ...)).
    beta_refresh_freq : int
        How often to recalculate beta vectors.
    spread_window : int
        Rolling window for spread mean/std.
    cash_start : float
        Starting cash.
    notional : float
        Notional to trade (per trade, split among coins).
    trade_freq : int
        Only use every trade_freq-th row.
    execution_delay : int
        Trade execution delay (in ticks).
    enter_zscore : float, optional
        Z-score threshold to enter a trade (default: 2.0).
    exit_zscore : float, optional
        Z-score threshold to exit a trade (default: 0.3).
    """

    # Prepare price columns
    price_cols = list(basket)
    n_coins = len(basket)

    # Try to infer ask/bid columns from basket names
    # If basket is ('bidPrice0_random_normal', ...) then ask is 'askPrice0_random_normal', etc.
    ask_cols = [col.replace('bidPrice0', 'askPrice0') if 'bidPrice0' in col else col.replace('bid', 'ask') for col in price_cols]
    bid_cols = [col.replace('askPrice0', 'bidPrice0') if 'askPrice0' in col else col.replace('ask', 'bid') for col in price_cols]

    # Only use every trade_freq-th row
    data_sub = data.iloc[::trade_freq].copy()
    data_sub.reset_index(drop=True, inplace=True)
    N = len(data_sub)

    # Use the timestampEvent column as the date column
    if 'timestampEvent' in data_sub.columns:
        date_col = data_sub['timestampEvent'].values
    else:
        date_col = data_sub.index.values

    # Precompute rolling windows for spread mean/std
    price_matrix = data_sub[price_cols].values

    # Precompute beta vectors at each refresh point
    beta_vectors_list = []
    beta_indices = []
    beta_dates = []
    for idx in range(beta_refresh_freq, N, beta_refresh_freq):
        coin_basket_matrix = price_matrix[idx-beta_refresh_freq:idx]
        johansen_result = coint_johansen(coin_basket_matrix, det_order=0, k_ar_diff=1)
        trace_stats = johansen_result.lr1
        cv_95 = johansen_result.cvt[:, 1]
        num_coints = np.sum(trace_stats > cv_95)
        if num_coints == 0:
            # fallback: use first vector
            num_coints = 1
        beta_vectors_list.append(johansen_result.evec[:, :num_coints])
        beta_indices.append(idx)
        beta_dates.append(date_col[idx])

    # Assign beta vector and last refresh date for each row
    beta_vectors_per_row = []
    last_beta_refresh_date_per_row = []
    current_beta = None
    current_beta_refresh_date = None
    beta_ptr = 0
    for i in range(N):
        if beta_ptr < len(beta_indices) and i >= beta_indices[beta_ptr]:
            current_beta = beta_vectors_list[beta_ptr]
            current_beta_refresh_date = beta_dates[beta_ptr]
            beta_ptr += 1
        beta_vectors_per_row.append(current_beta)
        last_beta_refresh_date_per_row.append(current_beta_refresh_date)

    # Vectorized spread calculation (using most recent beta)
    spreads = np.full(N, np.nan)
    spread_means = np.full(N, np.nan)
    spread_stds = np.full(N, np.nan)
    z_scores = np.full(N, np.nan)
    normalized_betas = np.full((N, n_coins), np.nan)
    notional_betas = np.full((N, n_coins), np.nan)

    for i in range(N):
        beta_vecs = beta_vectors_per_row[i]
        if beta_vecs is not None:
            beta = beta_vecs[:, 0]
            norm_beta = beta / np.linalg.norm(beta)
            normalized_betas[i] = norm_beta
            notional_betas[i] = notional * norm_beta
            # Spread at time i
            spreads[i] = price_matrix[i] @ beta
            # Rolling window for mean/std
            if i >= spread_window:
                window_mat = price_matrix[i-spread_window:i+1]
                window_spreads = window_mat @ beta
                spread_means[i] = window_spreads.mean()
                spread_stds[i] = window_spreads.std()
                z_scores[i] = (spreads[i] - spread_means[i]) / spread_stds[i]

    # Prepare execution price vectors
    ask_matrix = data_sub[ask_cols].values
    bid_matrix = data_sub[bid_cols].values

    # For each row, get execution price vector (with delay)
    exec_indices = np.clip(np.arange(N) + execution_delay, 0, N-1)
    ask_exec = ask_matrix[exec_indices]
    bid_exec = bid_matrix[exec_indices]

    # For each row, choose price vector based on notional_beta sign
    price_vectors = np.where(notional_betas > 0, ask_exec, bid_exec)

    # Build history DataFrame
    history_dict = {
        'index': np.arange(N),
        'timestampEvent': date_col,
        'last_beta_vector_refresh': last_beta_refresh_date_per_row,
        'spread': spreads,
        'spread_mean': spread_means,
        'spread_std': spread_stds,
        'z_score': z_scores,
    }
    # Add each price column to history_df
    for j, col in enumerate(price_cols):
        history_dict[col] = price_matrix[:, j]
    history_df = pd.DataFrame(history_dict)

    # --- Vectorized Trading Logic ---
    position = np.zeros(n_coins)
    long = False
    short = False
    cash = cash_start
    cash_vec = [cash]
    position_history = [position.copy()]
    trade_log = []

    # Track ticks between enters and exits
    ticks_since_entry = None  # None means not in a position

    for i in range(N):
        z = z_scores[i]
        if np.isnan(z):
            cash_vec.append(cash_vec[-1])
            position_history.append(position.copy())
            continue

        notional_beta = notional_betas[i]
        price_vector = price_vectors[i]

        # Compute before/after position value in cash (dot product)
        before_position_value = position @ price_vector

        # Exit long
        if z > -exit_zscore and long:
            ticks_since_entry += 1
            before_position = position.copy()
            before_cash = cash
            before_position_value = position @ price_vector
            cash = cash + before_position_value
            after_cash = cash
            position = np.zeros(n_coins)
            after_position = position.copy()
            after_position_value = position @ price_vector
            long = False
            trade_log.append({
                'step': i,
                'timestampEvent': date_col[i],
                'last_beta_vector_refresh': last_beta_refresh_date_per_row[i],
                'action': 'exit_long',
                'before_cash': before_cash,
                'after_cash': after_cash,
                'price_vector': price_vector.copy(),
                'before_position': before_position,
                'after_position': after_position,
                'before_position_value': before_position_value,
                'after_position_value': after_position_value,
                'z_score': z,
                'ticks_since_entry': ticks_since_entry
            })
            ticks_since_entry = None  # Reset after exit
        # Exit short
        elif z < exit_zscore and short:
            ticks_since_entry += 1
            before_position = position.copy()
            before_cash = cash
            before_position_value = position @ price_vector
            cash = cash + before_position_value
            after_cash = cash
            position = np.zeros(n_coins)
            after_position = position.copy()
            after_position_value = position @ price_vector
            short = False
            trade_log.append({
                'step': i,
                'timestampEvent': date_col[i],
                'last_beta_vector_refresh': last_beta_refresh_date_per_row[i],
                'action': 'exit_short',
                'before_cash': before_cash,
                'after_cash': after_cash,
                'price_vector': price_vector.copy(),
                'before_position': before_position,
                'after_position': after_position,
                'before_position_value': before_position_value,
                'after_position_value': after_position_value,
                'z_score': z,
                'ticks_since_entry': ticks_since_entry
            })
            ticks_since_entry = None  # Reset after exit
        # Enter/extend long
        elif z < -enter_zscore:
            before_position = position.copy()
            before_cash = cash
            before_position_value = position @ price_vector
            delta_position = notional_beta / price_vector
            position = position + delta_position
            after_position = position.copy()
            after_position_value = position @ price_vector
            cash = cash - delta_position @ price_vector
            after_cash = cash
            action = 'extend_long' if long else 'enter_long'
            trade_log.append({
                'step': i,
                'timestampEvent': date_col[i],
                'last_beta_vector_refresh': last_beta_refresh_date_per_row[i],
                'action': action,
                'before_cash': before_cash,
                'after_cash': after_cash,
                'price_vector': price_vector.copy(),
                'before_position': before_position,
                'after_position': after_position,
                'before_position_value': before_position_value,
                'after_position_value': after_position_value,
                'z_score': z,
                'ticks_since_entry': ticks_since_entry if long else 0
            })
            if long:
                if ticks_since_entry is not None:
                    ticks_since_entry += 1
            else:
                long = True
                ticks_since_entry = 0  # Start counting after entry
        # Enter/extend short
        elif z > enter_zscore:
            before_position = position.copy()
            before_cash = cash
            before_position_value = position @ price_vector
            delta_position = -notional_beta / price_vector
            position = position + delta_position
            after_position = position.copy()
            after_position_value = position @ price_vector
            cash = cash - delta_position @ price_vector
            after_cash = cash
            action = 'extend_short' if short else 'enter_short'
            trade_log.append({
                'step': i,
                'timestampEvent': date_col[i],
                'last_beta_vector_refresh': last_beta_refresh_date_per_row[i],
                'action': action,
                'before_cash': before_cash,
                'after_cash': after_cash,
                'price_vector': price_vector.copy(),
                'before_position': before_position,
                'after_position': after_position,
                'before_position_value': before_position_value,
                'after_position_value': after_position_value,
                'z_score': z,
                'ticks_since_entry': ticks_since_entry if short else 0
            })
            if short:
                if ticks_since_entry is not None:
                    ticks_since_entry += 1
            else:
                short = True
                ticks_since_entry = 0  # Start counting after entry
        else:
            # No trade
            if (long or short) and ticks_since_entry is not None:
                ticks_since_entry += 1

        cash_vec.append(cash)
        position_history.append(position.copy())

    # Final history as DataFrame
    history_df['cash'] = cash_vec[1:]
    history_df['position'] = position_history[1:]

    trade_log = pd.DataFrame(trade_log)

    return history_df, trade_log

def run_backtest_for_basket(
    basket,
    data,
    beta_refresh_freq,
    spread_window,
    cash_start,
    notional,
    trade_freq,
    execution_delay,
    enter_zscore,
    exit_zscore
):
    history_df, trade_log = vectorized_cointegrated_basket_backtest(
        data=data,
        basket=basket,
        beta_refresh_freq=beta_refresh_freq,
        spread_window=spread_window,
        cash_start=cash_start,
        notional=notional,
        trade_freq=trade_freq,
        execution_delay=execution_delay,
        enter_zscore=enter_zscore,
        exit_zscore=exit_zscore
    )
    print(f"Complete backtest of basket: {basket} with params: "
          f"beta_refresh_freq={beta_refresh_freq}, spread_window={spread_window}, "
          f"cash_start={cash_start}, notional={notional}, trade_freq={trade_freq}, "
          f"execution_delay={execution_delay}, enter_zscore={enter_zscore}, exit_zscore={exit_zscore}")
    import random
    # Return a dict of parameters for easier tracking
    # unique_id = f"{basket}_{random.randint(100000, 999999)}"
    unique_id = f"{basket}_{beta_refresh_freq}_{trade_freq}_{spread_window}_{execution_delay}"
    params = {
        "id": unique_id,
        "basket": basket,
        "beta_refresh_freq": beta_refresh_freq,
        "spread_window": spread_window,
        "cash_start": cash_start,
        "notional": notional,
        "trade_freq": trade_freq,
        "execution_delay": execution_delay,
        "enter_zscore": enter_zscore,
        "exit_zscore": exit_zscore
    }
    return params, history_df, trade_log

import itertools

def main(
    baskets,
    data,
    beta_refresh_freq=[1000],
    spread_window=[100],
    cash_start=[10000],
    notional=[100],
    trade_freq=[1],
    execution_delay=[0],
    enter_zscore=[2.0],
    exit_zscore=[0.3],
    use_multiprocessing=True
):
    """
    Run backtests for all combinations of parameter values.

    Parameters
    ----------
    baskets : list of tuple[str, ...]
        List of baskets to test.
    data : pd.DataFrame
        DataFrame with price data.
    beta_refresh_freq, spread_window, cash_start, notional, trade_freq, execution_delay, enter_zscore, exit_zscore :
        Each should be a list of values to try.
    use_multiprocessing : bool
        Whether to use multiprocessing.

    Returns
    -------
    results : list
        List of (params_dict, history_df, trade_log) tuples for each parameter combination.
    """

    # Ensure all parameters are lists
    param_lists = [
        baskets,
        beta_refresh_freq,
        spread_window,
        cash_start,
        notional,
        trade_freq,
        execution_delay,
        enter_zscore,
        exit_zscore
    ]

    # Generate all combinations
    combos = list(itertools.product(*param_lists))

    args = [
        (
            basket,
            data,
            beta_refresh_freq,
            spread_window,
            cash_start,
            notional,
            trade_freq,
            execution_delay,
            enter_zscore,
            exit_zscore
        )
        for (basket, beta_refresh_freq, spread_window, cash_start, notional, trade_freq, execution_delay, enter_zscore, exit_zscore) in combos
    ]

    if use_multiprocessing:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(run_backtest_for_basket, args)
    else:
        results = [run_backtest_for_basket(*arg) for arg in args]

    return results

if __name__ == "__main__":
    # This will only run if the script is run directly
    pass