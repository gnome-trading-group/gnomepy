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
    exit_zscore: float = 0.3,
    stop_loss_delta: float = 0.0,
    retest_cointegration: bool = False,
    use_extends: bool = True,
    use_lob: bool = True
):
    """
    Vectorized research_old for cointegrated basket trading.

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
    stop_loss_delta : float, optional
        Additional z-score distance for stop loss exit (default: 0.0).
    retest_cointegration : bool, optional
        If True, only trade when a cointegrating relationship is detected at each beta refresh.
        If False, continue trading even if no cointegration is detected (default: False).
    use_extends : bool, optional
        If True, allows extending positions based on z-score. If False, only allows enter/exit (default: True).
    use_lob : bool, optional
        If True, uses order book imbalance for trade signals (default: True).
    """

    # Prepare price columns
    price_cols = list(basket)
    n_coins = len(basket)

    # Try to infer ask/bid columns from basket names
    # If basket is ('bidPrice0_random_normal', ...) then ask is 'askPrice0_random_normal', etc.
    ask_cols = [col.replace('bidPrice0', 'askPrice0') if 'bidPrice0' in col else col.replace('bid', 'ask') for col in price_cols]
    bid_cols = [col.replace('askPrice0', 'bidPrice0') if 'askPrice0' in col else col.replace('ask', 'bid') for col in price_cols]

    # Get order book imbalance columns if use_lob is True
    if use_lob:
        imb_cols = [col.replace('bidPrice0', 'order_book_balance').replace('bid', 'order_book_balance') for col in price_cols]

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
    
    # Get imbalance matrix if use_lob is True
    if use_lob:
        imb_matrix = data_sub[imb_cols].values

    # Precompute beta vectors and number of cointegrating relationships at each refresh point
    beta_vectors_list = []
    beta_indices = []
    beta_dates = []
    num_coints_list = []
    for idx in range(beta_refresh_freq, N, beta_refresh_freq):
        coin_basket_matrix = price_matrix[idx-beta_refresh_freq:idx]
        johansen_result = coint_johansen(coin_basket_matrix, det_order=0, k_ar_diff=1)
        trace_stats = johansen_result.lr1
        cv_95 = johansen_result.cvt[:, 1]
        num_coints = np.sum(trace_stats > cv_95)
        if num_coints == 0:
            # No cointegrating relationship found, mark as None if retest_cointegration == True
            if retest_cointegration:
                beta_vectors_list.append(None)
            
            # If retest_cointegration == False, then we trade regardless
            else:
                num_coints = 1
                beta_vectors_list.append(johansen_result.evec[:, :num_coints])

        else:
            beta_vectors_list.append(johansen_result.evec[:, :num_coints])

        num_coints_list.append(num_coints)

        beta_indices.append(idx)
        beta_dates.append(date_col[idx])

    # Assign beta vector, last refresh date, and num_coints for each row
    beta_vectors_per_row = []
    last_beta_refresh_date_per_row = []
    num_coints_per_row = []
    current_beta = None
    current_beta_refresh_date = None
    current_num_coints = 0
    beta_ptr = 0
    for i in range(N):
        if beta_ptr < len(beta_indices) and i >= beta_indices[beta_ptr]:
            current_beta = beta_vectors_list[beta_ptr]
            current_beta_refresh_date = beta_dates[beta_ptr]
            current_num_coints = num_coints_list[beta_ptr]
            beta_ptr += 1
        beta_vectors_per_row.append(current_beta)
        last_beta_refresh_date_per_row.append(current_beta_refresh_date)
        num_coints_per_row.append(current_num_coints)

    # Vectorized spread calculation (using most recent beta)
    spreads = np.full(N, np.nan)
    spread_means = np.full(N, np.nan)
    spread_stds = np.full(N, np.nan)
    z_scores = np.full(N, np.nan)
    normalized_betas = np.full((N, n_coins), np.nan)
    notional_betas = np.full((N, n_coins), np.nan)

    for i in range(N):
        beta_vecs = beta_vectors_per_row[i]
        num_coints = num_coints_per_row[i]
        # Only compute spread/zscore if there is at least one cointegrating relationship
        if beta_vecs is not None and num_coints > 0:
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
        # If no cointegration, leave as NaN (do not trade)

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
        'num_coints': num_coints_per_row,
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
    extends_since_entry = None  # None means not in a position

    # stop_loss_delta is now an explicit argument

    last_beta_vector = None
    for i in range(N):
        z = z_scores[i]
        num_coints = num_coints_per_row[i]
        # Only trade if there is at least one cointegrating relationship
        if np.isnan(z) or num_coints == 0:
            cash_vec.append(cash_vec[-1])
            position_history.append(position.copy())
            continue

        notional_beta = notional_betas[i]
        price_vector = price_vectors[i]

        # Get imbalance if using LOB
        if use_lob:
            # Get imbalance for each security
            imb_vec = imb_matrix[i]
            # Check if imbalance aligns with beta direction for each security
            valid_lob = np.where(notional_beta > 0, imb_vec > 0.1, imb_vec < -0.1)
            # Only proceed if all securities have valid LOB signals
            lob_signal = np.all(valid_lob)
        else:
            lob_signal = True

        # Check for beta vector refresh
        beta_refresh = False
        if i == 0:
            last_beta_vector = last_beta_refresh_date_per_row[i]
        else:
            if last_beta_refresh_date_per_row[i] != last_beta_vector:
                beta_refresh = True
                last_beta_vector = last_beta_refresh_date_per_row[i]

        # Compute before/after position value in cash (dot product)
        before_position_value = position @ price_vector

        # --- Stop Loss Logic ---
        action = None
        before_position = None
        after_position = None
        after_cash = None
        scaled_notional = None
        
        # Exit long position (stop loss, normal exit, or beta refresh)
        if long and (z > enter_zscore + stop_loss_delta or z > -exit_zscore or beta_refresh):
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
            
            # Determine exit reason
            if z > enter_zscore + stop_loss_delta:
                action = 'stop_loss_exit_long'
            elif beta_refresh:
                action = 'beta_refresh_exit_long'
            else:
                action = 'exit_long'
            
        # Exit short position (stop loss, normal exit, or beta refresh)
        elif short and (z < -enter_zscore - stop_loss_delta or z < exit_zscore or beta_refresh):
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
            
            # Determine exit reason
            if z < -enter_zscore - stop_loss_delta:
                action = 'stop_loss_exit_short'
            elif beta_refresh:
                action = 'beta_refresh_exit_short'
            else:
                action = 'exit_short'
                
        # Enter/extend long (but not if stop loss would be triggered)
        elif z < -enter_zscore and (not use_lob or (use_lob and lob_signal)):
            # Only extend/enter if not past stop loss threshold
            if z > enter_zscore + stop_loss_delta:
                # Do not extend/enter, treat as no trade (should not happen, but for safety)
                if (long or short) and ticks_since_entry is not None:
                    ticks_since_entry += 1
            else:
                # Skip if already in position and extends not allowed
                if not (long and not use_extends):  # Changed to allow continuing execution
                    before_position = position.copy()
                    before_cash = cash
                    before_position_value = position @ price_vector
                    # Scale notional based on z-score magnitude
                    z_scale = min(abs(z / enter_zscore), 3.0)  # Cap at 3x
                    scaled_notional_beta = notional_beta * z_scale
                    scaled_notional = notional * z_scale
                    delta_position = scaled_notional_beta / price_vector
                    position = position + delta_position
                    after_position = position.copy()
                    after_position_value = position @ price_vector
                    cash = cash - delta_position @ price_vector
                    after_cash = cash
                    action = 'extend_long' if long else 'enter_long'

                    if long:
                        if ticks_since_entry is not None:
                            ticks_since_entry += 1
                            extends_since_entry += 1
                    else:
                        long = True
                        ticks_since_entry = 0  # Start counting after entry
                        extends_since_entry = 0  # Start counting after entry

                if (long or short) and ticks_since_entry is not None:
                    ticks_since_entry += 1

        # Enter/extend short (but not if stop loss would be triggered)
        elif z > enter_zscore and (not use_lob or (use_lob and lob_signal)):
            # Only extend/enter if not past stop loss threshold
            if z < -enter_zscore - stop_loss_delta:
                # Do not extend/enter, treat as no trade (should not happen, but for safety)
                if (long or short) and ticks_since_entry is not None:
                    ticks_since_entry += 1
            else:
                # Skip if already in position and extends not allowed
                if not (short and not use_extends):  # Changed to allow continuing execution
                    before_position = position.copy()
                    before_cash = cash
                    before_position_value = position @ price_vector
                    # Scale notional based on z-score magnitude
                    z_scale = min(abs(z / enter_zscore), 3.0)  # Cap at 3x
                    scaled_notional_beta = -notional_beta * z_scale
                    scaled_notional = notional * z_scale
                    delta_position = scaled_notional_beta / price_vector
                    position = position + delta_position
                    after_position = position.copy()
                    after_position_value = position @ price_vector
                    cash = cash - delta_position @ price_vector
                    after_cash = cash
                    action = 'extend_short' if short else 'enter_short'

                    if short:
                        if ticks_since_entry is not None:
                            ticks_since_entry += 1
                            extends_since_entry += 1
                    else:
                        short = True
                        ticks_since_entry = 0  # Start counting after entry
                        extends_since_entry = 0  # Start counting after entry

                if (long or short) and ticks_since_entry is not None:
                    ticks_since_entry += 1

        else:
            # No trade
            if (long or short) and ticks_since_entry is not None:
                ticks_since_entry += 1

        # Log trade if action was taken
        if action is not None:
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
                'ticks_since_entry': ticks_since_entry if ticks_since_entry is not None else 0,
                'extends_since_entry': extends_since_entry if extends_since_entry is not None else 0,
                'scaled_notional': scaled_notional
            })

            if 'exit' in action:
                ticks_since_entry = None  # Reset after exit
                extends_since_entry = None  # Reset after exit

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
    exit_zscore,
    stop_loss_delta=None,
    retest_cointegration=False,
    use_extends=True,
    use_lob=True
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
        exit_zscore=exit_zscore,
        stop_loss_delta=stop_loss_delta,
        retest_cointegration=retest_cointegration,
        use_extends=use_extends,
        use_lob=use_lob
    )
    # print(f"Complete research_old of basket: {basket} with params: "
    #       f"beta_refresh_freq={beta_refresh_freq}, spread_window={spread_window}, "
    #       f"cash_start={cash_start}, notional={notional}, trade_freq={trade_freq}, "
    #       f"execution_delay={execution_delay}, enter_zscore={enter_zscore}, exit_zscore={exit_zscore}, "
    #       f"stop_loss_delta={stop_loss_delta}, retest_cointegration={retest_cointegration}, "
    #       f"use_extends={use_extends}, use_lob={use_lob}")
    import random
    unique_id = (
        f"{basket}_"
        f"beta_refresh_freq={beta_refresh_freq}_"
        f"spread_window={spread_window}_"
        f"cash_start={cash_start}_"
        f"notional={notional}_"
        f"trade_freq={trade_freq}_"
        f"execution_delay={execution_delay}_"
        f"enter_zscore={enter_zscore}_"
        f"exit_zscore={exit_zscore}_"
        f"stop_loss_delta={stop_loss_delta}_"
        f"retest_cointegration={retest_cointegration}_"
        f"use_extends={use_extends}_"
        f"use_lob={use_lob}"
    )
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
        "exit_zscore": exit_zscore,
        "stop_loss_delta": stop_loss_delta,
        "retest_cointegration": retest_cointegration,
        "use_extends": use_extends,
        "use_lob": use_lob
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
    stop_loss_delta=[0],
    retest_cointegration=[False],
    use_extends=[True],
    use_lob=[True],
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
    beta_refresh_freq, spread_window, cash_start, notional, trade_freq, execution_delay, enter_zscore, exit_zscore, stop_loss_delta, retest_cointegration, use_extends, use_lob :
        Each should be a list of values to try.
    use_multiprocessing : bool
        Whether to use multiprocessing.

    Returns
    -------
    results : list
        List of (params_dict, history_df, trade_log) tuples for each parameter combination.
    """
    from tqdm import tqdm

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
        exit_zscore,
        stop_loss_delta,
        retest_cointegration,
        use_extends,
        use_lob
    ]

    # Generate all combinations
    combos = list(itertools.product(*param_lists))
    total_combos = len(combos)
    print(f"Running {total_combos} parameter combinations...")

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
            exit_zscore,
            stop_loss_delta,
            retest_cointegration,
            use_extends,
            use_lob
        )
        for (basket, beta_refresh_freq, spread_window, cash_start, notional, trade_freq, execution_delay, enter_zscore, exit_zscore, stop_loss_delta, retest_cointegration, use_extends, use_lob) in combos
    ]

    if use_multiprocessing:
        with mp.Pool(processes=4) as pool:
            results = list(tqdm(pool.imap(lambda x: run_backtest_for_basket(*x), args), total=total_combos))
    else:
        results = []
        for arg in tqdm(args, total=total_combos):
            results.append(run_backtest_for_basket(*arg))

    return results

if __name__ == "__main__":
    # This will only run if the script is run directly
    pass