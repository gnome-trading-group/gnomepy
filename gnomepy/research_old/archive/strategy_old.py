from gnomepy.research_old.backtest import *
from gnomepy.research_old.archive.strategy_old import *
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import multiprocessing as mp
from gnomepy.research_old.archive.coint_testing import *
from IPython.display import display
import pandas as pd
import datetime

class Strategy:

    def __init__(self, basket: list[tuple[int, ...]], data_schema_type: SchemaType = SchemaType.MBP_10,
                 trade_frequency: int = 1):
        """Initialize base strategy class"""
        self.basket = basket
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency

    def process_trade(self, lob_data: pd.DataFrame, action: str, order_type: str, size: float, exchange: int, security: int, limit_price: float = None):
        """Process a trade based on limit order book data.
        
        Args:
            lob_data (pd.DataFrame): Limit order book data
            action (str): Trade action - either 'buy' or 'sell'
            order_type (str): Order type - either 'market' or 'limit'
            size (float): Number of units to trade
            exchange (int): Exchange ID
            security (int): Security ID
            limit_price (float, optional): Price for limit orders. Required if order_type is 'limit'.
        """
        remaining_size = size
        filled_size = 0
        weighted_price = 0
        
        # Track fills at each level
        level_fills = []
        
        # Look through order book levels until we fill the full size
        for level in range(10):  # Assuming 10 levels in the order book
            if action == 'buy':
                price = lob_data[f'askPrice{level}_exchange{exchange}_security{security}'] if order_type == 'market' else limit_price
                available_size = lob_data[f'askSize{level}_exchange{exchange}_security{security}']

            else:  # sell
                price = lob_data[f'bidPrice{level}_exchange{exchange}_security{security}'] if order_type == 'market' else limit_price
                available_size = lob_data[f'bidSize{level}_exchange{exchange}_security{security}']


            # Skip if no size available at this level
            if available_size.item() <= 0:
                continue
                
            # Randomly reduce available size to simulate competition
            # We can get between 30% to 90% of the displayed size
            competition_factor = np.random.uniform(0.6, 1.0)
            available_size = available_size.item() * competition_factor
                
            # Calculate how much we can fill at this level
            fill_size = min(remaining_size, available_size)
            filled_size += fill_size
            weighted_price += price.item() * fill_size
            remaining_size -= fill_size
                        
            # Record fill at this level
            level_fills.append({
                'level': level,
                'price': price.item(),
                'size': fill_size,
                'pct_of_total': fill_size / size
            })
            
            # Break if we've filled the entire order
            if remaining_size <= 0:
                break

        # Return unfilled if we couldn't fill any size
        if filled_size == 0:
            return {
                'filled': False,
                'price': None,
                'size': 0,
                'cash': 0,
                'level_fills': []
            }
        
        # Calculate total cash with correct sign based on action and add fees
        total_cash = weighted_price
        fee = total_cash * 4.5e-4  # Calculate fee as 0.045% of trade value
        
        if action == 'buy':
            total_cash = -(total_cash + fee)  # Negative for buys, add fee
        else:  # sell
            total_cash = total_cash - fee  # Positive for sells, subtract fee
        
        # Calculate percent filled at level 0
        level0_fill_pct = 0
        if level_fills and level_fills[0]['level'] == 0:
            level0_fill_pct = level_fills[0]['pct_of_total']
            
        # Return partially/fully filled order details
        return {
            'filled': True,
            'price': weighted_price / filled_size,  # Calculate volume-weighted average price
            'size': filled_size,
            'cash': total_cash,  # Total cash including fees (negative for buys, positive for sells)
            'level_fills': level_fills,  # Details about fills at each price level
            'level0_fill_pct': level0_fill_pct  # Percent of total order filled at level 0
        }

    def process_backtest(self, history_df, trade_log, params):
        """Process research_old results and compute summary statistics.
        
        Parameters
        ----------
        history_df : pd.DataFrame
            Historical price data used in research_old
        trade_log : pd.DataFrame 
            Log of all trades executed during research_old
        params : dict
            Dictionary of strategy parameters
            
        Returns
        -------
        dict
            Summary statistics of research_old performance including:
            - Number of complete trades
            - Average profit per trade
            - Standard deviation of profits
            - Total profit
            - Average ticks per trade
            - Win ratio
            - Max drawdown
            - Profit factor
            - Sharpe ratio
        pd.DataFrame
            Updated trade log with P&L calculations
        """
        if trade_log.shape[0] == 0:
            summary = {
                'num_complete_trades': -1,
                'avg_profit_per_complete_trade': -1,
                'std_profit_per_complete_trade': -1,
                'total_profit': -1,
                'avg_ticks_per_complete_trade': -1,
                'std_ticks_per_complete_trade': -1,
                'win_ratio': -1,
                'max_drawdown': -1,
                'profit_factor': -1,
                'sharpe_ratio': -1
            }
            # Add params to summary
            for k, v in params.items():
                summary[k] = v
            return summary, trade_log

        else:
            trade_log['cash_delta'] = trade_log['after_cash'] - trade_log['before_cash']

            # Only consider enter and exit trades for PL calculation, ignore extend trades
            enter_exit_mask = trade_log['action'].str.contains('enter|exit')
            enter_exit_trades = trade_log[enter_exit_mask].reset_index(drop=True)

            # Find indices of exit trades
            exit_mask = enter_exit_trades['action'].str.contains('exit')
            exit_indices = enter_exit_trades.index[exit_mask].tolist()

            # For each exit, find the immediately preceding enter (assume always alternates)
            pl_list, ticks_list, extends_list = [], [], []
             
            for exit_idx in exit_indices:
                # The enter trade is always the previous row before exit in enter_exit_trades
                enter_idx = exit_idx - 1
                if enter_idx >= 0 and 'enter' in enter_exit_trades.loc[enter_idx, 'action']:
                    # Get step indices for this complete trade
                    enter_step = enter_exit_trades.loc[enter_idx, 'step']
                    exit_step = enter_exit_trades.loc[exit_idx, 'step']
                    
                    # Get all trades (enter, extends, exit) between these steps
                    trade_slice = trade_log[(trade_log['step'] >= enter_step) & (trade_log['step'] <= exit_step)]
                    
                    # Calculate P&L 
                    pl = enter_exit_trades.loc[exit_idx, 'after_cash'] - enter_exit_trades.loc[enter_idx, 'before_cash']
                    pl_list.append(pl)
                    
                    ticks_list.append(enter_exit_trades.loc[exit_idx, 'ticks_since_entry'])
                    extends_list.append(enter_exit_trades.loc[exit_idx, 'extends_since_entry'])

                    # Update trade_log with PL values for this exit trade
                    trade_log.loc[trade_log['step'] == exit_step, 'pl'] = pl
                else:
                    # If for some reason the previous is not an enter, skip
                    continue

            # Create a DataFrame for exit trades with calculated PL and ticks
            exit_trades = pd.DataFrame({
                'pl': pl_list,
                'ticks_since_entry': ticks_list,
                'extends_since_entry': extends_list
            })

            # Calculate max drawdown
            if 'after_cash' in trade_log.columns and len(exit_trades) > 0:
                exit_trades_cumsum = exit_trades['pl'].cumsum()
                running_max = pd.Series(exit_trades_cumsum).cummax()
                drawdown = exit_trades_cumsum - running_max
                max_drawdown = drawdown.min()
            else:
                max_drawdown = np.nan

            # Calculate Sharpe ratio (using per-trade P&L)
            if len(exit_trades) > 1:
                mean_pl = exit_trades['pl'].mean()
                std_pl = exit_trades['pl'].std()
                days_in_dataset = (history_df['timestampEvent'].max() - history_df['timestampEvent'].min()).total_seconds() / (24 * 60 * 60)
                sharpe_ratio = mean_pl / std_pl * np.sqrt(365 / days_in_dataset) if std_pl != 0 and days_in_dataset > 0 else np.nan
            else:
                sharpe_ratio = np.nan

            summary = {
                'num_complete_trades': len(exit_trades),
                'avg_profit_per_complete_trade': exit_trades['pl'].mean(),
                'std_profit_per_complete_trade': exit_trades['pl'].std(),
                'total_profit': exit_trades['pl'].sum(),
                'avg_ticks_per_complete_trade': exit_trades['ticks_since_entry'].mean(),
                'std_ticks_per_complete_trade': exit_trades['ticks_since_entry'].std(),
                'avg_extends_per_complete_trade': exit_trades['extends_since_entry'].mean(),
                'std_extends_per_complete_trade': exit_trades['extends_since_entry'].std(),
                'win_ratio': exit_trades[exit_trades['pl'] > 0].shape[0] / len(exit_trades),
                'max_drawdown': max_drawdown,
                'profit_factor': exit_trades[exit_trades['pl'] > 0]['pl'].sum() / np.abs(exit_trades[exit_trades['pl'] < 0]['pl'].sum()),
                'sharpe_ratio': sharpe_ratio,
                'profit_to_drawdown': exit_trades['pl'].sum() / abs(max_drawdown) if max_drawdown != 0 else np.nan
            }
            # Add params to summary
            for k, v in params.items():
                summary[k] = v

            return summary, trade_log

        

class CointegrationStrategy(Strategy):

    def __init__(self,basket: list[tuple[int, ...]], data_schema_type: SchemaType = SchemaType.MBP_10,
                 trade_frequency: int = 1, beta_refresh_frequency: int = 1000, 
                 spread_window: int = 100, enter_zscore: float = 2.0, exit_zscore: float = 0.3, 
                 stop_loss_delta: float = 0.0, retest_cointegration: bool = False, use_extends: bool = True,
                 use_lob: bool = True, use_dynamic_sizing: bool = True):
        """Initialize a cointegration trading strategy.
        
        Parameters
        ----------
        basket : list[tuple[int, ...]]
            List of tuples containing exchange and security IDs to trade as a cointegrated basket
        data_schema_type : SchemaType, default SchemaType.MBP_10
            Type of market data schema to use
        trade_frequency : int, default 1
            How frequently to check for trading signals
        beta_refresh_frequency : int, default 1000
            How frequently to recalculate cointegration betas
        spread_window : int, default 100
            Rolling window size for calculating spread statistics
        enter_zscore : float, default 2.0
            Z-score threshold to enter positions
        exit_zscore : float, default 0.3
            Z-score threshold to exit positions
        stop_loss_delta : float, default 0
            Stop loss threshold, 0 means no stop loss
        retest_cointegration : bool, default False
            Whether to retest cointegration at beta refresh frequency level
        use_extends : bool, default True
            Whether to allow extending positions
        use_lob : bool, default True
            Whether to use limit order book data
        use_dynamic_sizing : bool, default False
            Whether to use dynamic position sizing
        """
        super().__init__(basket, data_schema_type, trade_frequency)

        self.beta_refresh_frequency = beta_refresh_frequency
        self.spread_window = spread_window
        self.enter_zscore = enter_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_delta = stop_loss_delta
        self.retest_cointegration = retest_cointegration
        self.use_extends = use_extends
        self.use_lob = use_lob
        self.use_dynamic_sizing = use_dynamic_sizing


    def retrieve_data(self, start_datetime: datetime.datetime, end_datetime: datetime.datetime):
        """Retrieve and preprocess market data for the cointegration strategy.

        This function fetches market data from AWS S3 for each security in the basket,
        resamples it according to trade frequency, renames columns with exchange/security
        identifiers, calculates order book imbalance metrics, and merges data from all
        securities and exchanges into a single DataFrame using asof joins to handle
        non-aligned timestamps across different venues.

        Parameters
        ----------
        start_datetime : datetime.datetime
            Start time for retrieving market data
        end_datetime : datetime.datetime
            End time for retrieving market data

        Returns
        -------
        pd.DataFrame
            Merged DataFrame containing processed market data for all securities
            and exchanges, aligned by nearest timestamp
        """
        
        dfs = []

        # Calculate order book balance for this security
        bid_size_cols = [f'bidSize{i}' for i in range(10)]
        ask_size_cols = [f'askSize{i}' for i in range(10)]
        bid_price_cols = [f'bidPrice{i}' for i in range(10)]
        ask_price_cols = [f'askPrice{i}' for i in range(10)]

        for item in self.basket:
            # Get exchange and security of basket item
            exchange, security = item[0], item[1]

            # Get data from AWS S3
            client = MarketDataClient(bucket="gnome-market-data-prod", aws_profile_name="AWSAdministratorAccess-241533121172")
            client_data_params = {
                "exchange_id": security, # this is currently flipped
                "security_id": exchange, # with this
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "schema_type": self.data_schema_type,
            }
            df = client.get_data(**client_data_params).to_df()

            # Immediately resample the data for faster computation / lower mem footprint
            df = df[::self.trade_frequency]
            df.reset_index(drop=True, inplace=True)

            # Sum up total bid and ask sizes
            total_bid_size = df[bid_size_cols].sum(axis=1)
            total_ask_size = df[ask_size_cols].sum(axis=1)
            
            # Calculate and normalize order book balance
            order_book_balance = total_bid_size - total_ask_size
            df[f'order_book_balance_exchange{exchange}_security{security}'] = (order_book_balance - order_book_balance.mean()) / order_book_balance.std()

            # Rename all bid/ask price columns with exchange and security suffix
            for i in range(10):
                df = df.rename(columns={
                    f'bidPrice{i}': f'bidPrice{i}_exchange{exchange}_security{security}',
                    f'askPrice{i}': f'askPrice{i}_exchange{exchange}_security{security}',
                    f'bidSize{i}': f'bidSize{i}_exchange{exchange}_security{security}', 
                    f'askSize{i}': f'askSize{i}_exchange{exchange}_security{security}'
                })

            df = df.sort_values('timestampEvent')
            dfs.append(df)

        # Get all columns with price/size data for each exchange/security pair
        for i in range(1, len(dfs)):
            exchange, security = self.basket[i][0], self.basket[i][1]
            prev_exchange, prev_security = self.basket[i-1][0], self.basket[i-1][1]
            
            # Build list of columns to keep for merging
            merge_cols = ['timestampEvent']
            for j in range(10):
                merge_cols.extend([
                    f'bidPrice{j}_exchange{exchange}_security{security}',
                    f'askPrice{j}_exchange{exchange}_security{security}',
                    f'bidSize{j}_exchange{exchange}_security{security}',
                    f'askSize{j}_exchange{exchange}_security{security}'
                ])
            merge_cols.append(f'order_book_balance_exchange{exchange}_security{security}')

            if i == 1:
                merged_df = pd.merge_asof(
                    dfs[0],
                    dfs[i][merge_cols],
                    on='timestampEvent',
                    direction='nearest'
                )
            else:
                merged_df = pd.merge_asof(
                    merged_df,
                    dfs[i][merge_cols],
                    on='timestampEvent',
                    direction='nearest'
                )

        return merged_df

    # def research_old(self, notional: float, cash_start: float,
    #              start_datetime: datetime = None, end_datetime: datetime = None):
    #     """
    #     Runs a research_old of the statistical arbitrage strategy.

    #     Args:
    #         notional (float): The base notional amount to trade with. This will be scaled based on z-score magnitude.
    #         cash_start (float): Initial cash balance to start trading with.
    #         start_datetime (datetime, optional): Start date/time to research_old from. If None, uses earliest available data.
    #         end_datetime (datetime, optional): End date/time to research_old until. If None, uses latest available data.

    #     Returns:
    #         tuple: A tuple containing:
    #             - history_df (pd.DataFrame): DataFrame containing the full trading history including:
    #                 - Timestamps, spreads, z-scores, positions, cash balances
    #                 - Order book data for each security
    #             - trade_log (pd.DataFrame): Detailed log of all trades executed including:
    #                 - Entry/exit points, position sizes, cash changes
    #                 - Trade metadata like z-scores and time in position
    #     """
        
    #     # First load in data
    #     data = self.retrieve_data(start_datetime=start_datetime, end_datetime=end_datetime)
    #     N = len(data)
    #     n_items = len(self.basket)

    #     # Configure some variables
    #     ask_price_cols = [f"askPrice{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
    #     bid_price_cols = [f"bidPrice{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
    #     ask_size_cols = [f"askSize{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
    #     bid_size_cols = [f"bidSize{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
    #     coint_bid_price_cols = [f"bidPrice0_exchange{item[0]}_security{item[1]}" for item in self.basket] # Simply use the bidPrice0 for cointegration and beta calcs
    #     coint_ask_price_cols = [f"askPrice0_exchange{item[0]}_security{item[1]}" for item in self.basket] # Simply use the bidPrice0 for cointegration and beta calcs
    #     lob_imbalance_cols = [f"order_book_balance_exchange{item[0]}_security{item[1]}" for item in self.basket]
    #     order_book_cols = ask_price_cols + bid_price_cols + ask_size_cols + bid_size_cols

    #     # Separate matrices
    #     date_matrix = data['timestampEvent'].values
    #     coint_price_matrix = data[coint_bid_price_cols].values
    #     lob_imbalance_matrix = data[lob_imbalance_cols].values
    #     bid_matrix = data[coint_bid_price_cols].values
    #     ask_matrix = data[coint_ask_price_cols].values
    #     order_book_matrix = np.hstack([
    #         data['timestampEvent'].values.astype('str').reshape(-1, 1),
    #         data[ask_price_cols].values.astype(float),
    #         data[bid_price_cols].values.astype(float),
    #         data[ask_size_cols].values.astype(float), 
    #         data[bid_size_cols].values.astype(float)
    #     ])

    #     # Precompute beta vectors and number of cointegrating relationships at each refresh point
    #     beta_vectors_list = []
    #     beta_indices = []
    #     beta_dates = []
    #     num_coints_list = []
    #     for idx in range(self.beta_refresh_frequency, N, self.beta_refresh_frequency):
    #         coin_basket_matrix = bid_matrix[idx-self.beta_refresh_frequency:idx]
    #         johansen_result = coint_johansen(coin_basket_matrix, det_order=0, k_ar_diff=1)
    #         trace_stats = johansen_result.lr1
    #         cv_95 = johansen_result.cvt[:, 1]
    #         num_coints = np.sum(trace_stats > cv_95)
    #         if num_coints == 0:
    #             # No cointegrating relationship found, mark as None if retest_cointegration == True
    #             if self.retest_cointegration:
    #                 beta_vectors_list.append(None)
                
    #             # If retest_cointegration == False, then we trade regardless
    #             else:
    #                 num_coints = 1
    #                 beta_vectors_list.append(johansen_result.evec[:, :num_coints])

    #         else:
    #             beta_vectors_list.append(johansen_result.evec[:, :num_coints])

    #         num_coints_list.append(num_coints)

    #         beta_indices.append(idx)
    #         beta_dates.append(date_matrix[idx])

    #     # Assign beta vector, last refresh date, and num_coints for each row
    #     beta_vectors_per_row = []
    #     last_beta_refresh_date_per_row = []
    #     num_coints_per_row = []
    #     current_beta = None
    #     current_beta_refresh_date = None
    #     current_num_coints = 0
    #     beta_ptr = 0
    #     for i in range(N):
    #         if beta_ptr < len(beta_indices) and i >= beta_indices[beta_ptr]:
    #             current_beta = beta_vectors_list[beta_ptr]
    #             current_beta_refresh_date = beta_dates[beta_ptr]
    #             current_num_coints = num_coints_list[beta_ptr]
    #             beta_ptr += 1
    #         beta_vectors_per_row.append(current_beta)
    #         last_beta_refresh_date_per_row.append(current_beta_refresh_date)
    #         num_coints_per_row.append(current_num_coints)


    #     # Vectorized spread calculation (using most recent beta)
    #     spreads = np.full(N, np.nan)
    #     spread_means = np.full(N, np.nan)
    #     spread_stds = np.full(N, np.nan)
    #     z_scores = np.full(N, np.nan)
    #     normalized_betas = np.full((N, n_items), np.nan)
    #     notional_betas = np.full((N, n_items), np.nan)

    #     for i in range(N):
    #         beta_vecs = beta_vectors_per_row[i]
    #         num_coints = num_coints_per_row[i]
    #         # Only compute spread/zscore if there is at least one cointegrating relationship
    #         if beta_vecs is not None and num_coints > 0:
    #             beta = beta_vecs[:, 0]
    #             norm_beta = beta / np.linalg.norm(beta)
    #             normalized_betas[i] = norm_beta
    #             notional_betas[i] = notional * norm_beta
    #             # Spread at time i
    #             spreads[i] = coint_price_matrix[i] @ beta
    #             # Rolling window for mean/std
    #             if i >= self.spread_window:
    #                 window_mat = coint_price_matrix[i-self.spread_window:i+1]
    #                 window_spreads = window_mat @ beta
    #                 spread_means[i] = window_spreads.mean()
    #                 spread_stds[i] = window_spreads.std()
    #                 z_scores[i] = (spreads[i] - spread_means[i]) / spread_stds[i]

    #     # For each row, get execution price vector (with delay)
    #     exec_indices = np.clip(np.arange(N), 0, N-1)
    #     ask_exec = ask_matrix[exec_indices]
    #     bid_exec = bid_matrix[exec_indices]

    #     # For each row, choose price vector based on notional_beta sign
    #     price_vectors = np.where(notional_betas > 0, ask_exec, bid_exec)
        
    #    # Build history DataFrame
    #     history_dict = {
    #         'index': np.arange(N),
    #         'timestampEvent': date_matrix,
    #         'last_beta_vector_refresh': last_beta_refresh_date_per_row,
    #         'spread': spreads,
    #         'spread_mean': spread_means,
    #         'spread_std': spread_stds,
    #         'z_score': z_scores,
    #         'num_coints': num_coints_per_row,
    #     }

    #     # Add each order book column to history_df    
    #     for j, col in enumerate(order_book_cols):
    #         history_dict[col] = order_book_matrix[:, j]

    #     history_df = pd.DataFrame(history_dict)

    #     # --- Vectorized Trading Logic ---
    #     position = np.zeros(n_items)
    #     long = False
    #     short = False
    #     cash = cash_start
    #     cash_vec = [cash]
    #     position_history = [position.copy()]
    #     trade_log = []

    #     # Track ticks between enters and exits
    #     ticks_since_entry = None  # None means not in a position
    #     extends_since_entry = None  # None means not in a position
    #     last_beta_vector = None

    #     # Iterate through each data point
    #     for i in range(N):
    #         z = z_scores[i]
    #         num_coints = num_coints_per_row[i]
    #         # Only trade if there is at least one cointegrating relationship
    #         if np.isnan(z) or num_coints == 0:
    #             cash_vec.append(cash_vec[-1])
    #             position_history.append(position.copy())
    #             continue

    #         notional_beta = notional_betas[i]
    #         price_vector = price_vectors[i]

    #         # Get imbalance if using LOB
    #         if self.use_lob:
    #             # Get imbalance for each security
    #             imb_vec = lob_imbalance_matrix[i]

    #             # Check if imbalance aligns with beta direction for each security
    #             valid_lob = np.where(notional_beta > 0, imb_vec > 0.1, imb_vec < -0.1)

    #             # Only proceed if all securities have valid LOB signals
    #             lob_signal = np.all(valid_lob)

    #         else:
    #             lob_signal = True

    #         # Check for beta vector refresh
    #         beta_refresh = False
    #         if i == 0:
    #             last_beta_vector = last_beta_refresh_date_per_row[i]
    #         else:
    #             if last_beta_refresh_date_per_row[i] != last_beta_vector:
    #                 beta_refresh = True
    #                 last_beta_vector = last_beta_refresh_date_per_row[i]

    #         # Compute before/after position value in cash (dot product)
    #         before_position_value = position @ price_vector

    #         # --- Stop Loss Logic ---
    #         action = None
    #         before_position = None
    #         after_position = None
    #         after_cash = None
    #         scaled_notional = None


    #         # Exit long position (stop loss, normal exit, or beta refresh)
    #         if long and (z > self.enter_zscore + self.stop_loss_delta or z > -self.exit_zscore or beta_refresh):
    #             ticks_since_entry += 1
    #             before_position = position.copy()
    #             before_cash = cash
    #             print(cash)
    #             print(position)
    #             print(price_vector)
    #             before_position_value = position @ price_vector
    #             print(before_position_value)
    #             cash = cash + before_position_value
    #             after_cash = cash
    #             position = np.zeros(n_items)
    #             after_position = position.copy()
    #             after_position_value = position @ price_vector
    #             long = False
                
    #             # Determine exit reason
    #             if z > self.enter_zscore + self.stop_loss_delta:
    #                 action = 'stop_loss_exit_long'
    #             elif beta_refresh:
    #                 action = 'beta_refresh_exit_long'
    #             else:
    #                 action = 'exit_long'
                
    #         # Exit short position (stop loss, normal exit, or beta refresh)
    #         elif short and (z < -self.enter_zscore - self.stop_loss_delta or z < self.exit_zscore or beta_refresh):
    #             ticks_since_entry += 1
    #             before_position = position.copy()
    #             before_cash = cash
    #             before_position_value = position @ price_vector
    #             cash = cash + before_position_value
    #             after_cash = cash
    #             position = np.zeros(n_items)
    #             after_position = position.copy()
    #             after_position_value = position @ price_vector
    #             short = False
                
    #             # Determine exit reason
    #             if z < -self.enter_zscore - self.stop_loss_delta:
    #                 action = 'stop_loss_exit_short'
    #             elif beta_refresh:
    #                 action = 'beta_refresh_exit_short'
    #             else:
    #                 action = 'exit_short'
                    
    #         # Enter/extend long (but not if stop loss would be triggered)
    #         elif z < -self.enter_zscore and (not self.use_lob or (self.use_lob and lob_signal)):
    #             # Only extend/enter if not past stop loss threshold
    #             if z > self.enter_zscore + self.stop_loss_delta:
    #                 # Do not extend/enter, treat as no trade (should not happen, but for safety)
    #                 if (long or short) and ticks_since_entry is not None:
    #                     ticks_since_entry += 1
    #             else:
    #                 # Skip if already in position and extends not allowed
    #                 if not (long and not self.use_extends):  # Changed to allow continuing execution
    #                     before_position = position.copy()
    #                     before_cash = cash
    #                     before_position_value = position @ price_vector
    #                     # Scale notional based on z-score magnitude
    #                     z_scale = min(abs(z / self.enter_zscore), 3.0)  # Cap at 3x
    #                     scaled_notional_beta = notional_beta * z_scale
    #                     scaled_notional = notional * z_scale
    #                     delta_position = scaled_notional_beta / price_vector
    #                     position = position + delta_position
    #                     after_position = position.copy()
    #                     after_position_value = position @ price_vector
    #                     cash = cash - delta_position @ price_vector
    #                     after_cash = cash
    #                     action = 'extend_long' if long else 'enter_long'

    #                     if long:
    #                         if ticks_since_entry is not None:
    #                             ticks_since_entry += 1
    #                             extends_since_entry += 1
    #                     else:
    #                         long = True
    #                         ticks_since_entry = 0  # Start counting after entry
    #                         extends_since_entry = 0  # Start counting after entry

    #                 if (long or short) and ticks_since_entry is not None:
    #                     ticks_since_entry += 1

    #         # Enter/extend short (but not if stop loss would be triggered)
    #         elif z > self.enter_zscore and (not self.use_lob or (self.use_lob and lob_signal)):
    #             # Only extend/enter if not past stop loss threshold
    #             if z < -self.enter_zscore - self.stop_loss_delta:
    #                 # Do not extend/enter, treat as no trade (should not happen, but for safety)
    #                 if (long or short) and ticks_since_entry is not None:
    #                     ticks_since_entry += 1
    #             else:
    #                 # Skip if already in position and extends not allowed
    #                 if not (short and not self.use_extends):  # Changed to allow continuing execution
    #                     before_position = position.copy()
    #                     before_cash = cash
    #                     before_position_value = position @ price_vector
    #                     # Scale notional based on z-score magnitude
    #                     z_scale = min(abs(z / self.enter_zscore), 3.0)  # Cap at 3x
    #                     scaled_notional_beta = -notional_beta * z_scale
    #                     scaled_notional = notional * z_scale
    #                     delta_position = scaled_notional_beta / price_vector
    #                     position = position + delta_position
    #                     after_position = position.copy()
    #                     after_position_value = position @ price_vector
    #                     cash = cash - delta_position @ price_vector
    #                     after_cash = cash
    #                     action = 'extend_short' if short else 'enter_short'

    #                     if short:
    #                         if ticks_since_entry is not None:
    #                             ticks_since_entry += 1
    #                             extends_since_entry += 1
    #                     else:
    #                         short = True
    #                         ticks_since_entry = 0  # Start counting after entry
    #                         extends_since_entry = 0  # Start counting after entry

    #                 if (long or short) and ticks_since_entry is not None:
    #                     ticks_since_entry += 1

    #         else:
    #             # No trade
    #             if (long or short) and ticks_since_entry is not None:
    #                 ticks_since_entry += 1

    #         # Log trade if action was taken
    #         if action is not None:
    #             trade_log.append({
    #                 'step': i,
    #                 'timestampEvent': date_matrix[i],
    #                 'last_beta_vector_refresh': last_beta_refresh_date_per_row[i],
    #                 'action': action,
    #                 'before_cash': before_cash,
    #                 'after_cash': after_cash,
    #                 'price_vector': price_vector.copy(),
    #                 'before_position': before_position,
    #                 'after_position': after_position,
    #                 'before_position_value': before_position_value,
    #                 'after_position_value': after_position_value,
    #                 'z_score': z,
    #                 'ticks_since_entry': ticks_since_entry if ticks_since_entry is not None else 0,
    #                 'extends_since_entry': extends_since_entry if extends_since_entry is not None else 0,
    #                 'scaled_notional': scaled_notional
    #             })

    #             if 'exit' in action:
    #                 ticks_since_entry = None  # Reset after exit
    #                 extends_since_entry = None  # Reset after exit

    #         cash_vec.append(cash)
    #         position_history.append(position.copy())

    #     # Final history as DataFrame
    #     history_df['cash'] = cash_vec[1:]
    #     history_df['position'] = position_history[1:]

    #     trade_log = pd.DataFrame(trade_log)

    #     return history_df, trade_log



    def backtest(self, notional: float, cash_start: float, 
                 start_datetime: datetime = None, end_datetime: datetime = None):
        """
        Runs a research_old of the statistical arbitrage strategy.

        Args:
            notional (float): The base notional amount to trade with. This will be scaled based on z-score magnitude.
            cash_start (float): Initial cash balance to start trading with.
            start_datetime (datetime, optional): Start date/time to research_old from. If None, uses earliest available data.
            end_datetime (datetime, optional): End date/time to research_old until. If None, uses latest available data.

        Returns:
            tuple: A tuple containing:
                - history_df (pd.DataFrame): DataFrame containing the full trading history including:
                    - Timestamps, spreads, z-scores, positions, cash balances
                    - Order book data for each security
                - trade_log (pd.DataFrame): Detailed log of all trades executed including:
                    - Entry/exit points, position sizes, cash changes
                    - Trade metadata like z-scores and time in position
        """
        
        # First load in data
        data = self.retrieve_data(start_datetime=start_datetime, end_datetime=end_datetime)
        N = len(data)
        n_items = len(self.basket)

        # Configure some variables
        ask_price_cols = [f"askPrice{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
        bid_price_cols = [f"bidPrice{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
        ask_size_cols = [f"askSize{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
        bid_size_cols = [f"bidSize{j}_exchange{item[0]}_security{item[1]}" for j in range(10) for item in self.basket]
        coint_bid_price_cols = [f"bidPrice0_exchange{item[0]}_security{item[1]}" for item in self.basket] # Simply use the bidPrice0 for cointegration and beta calcs
        coint_ask_price_cols = [f"askPrice0_exchange{item[0]}_security{item[1]}" for item in self.basket] # Simply use the bidPrice0 for cointegration and beta calcs
        lob_imbalance_cols = [f"order_book_balance_exchange{item[0]}_security{item[1]}" for item in self.basket]
        order_book_cols = ask_price_cols + bid_price_cols + ask_size_cols + bid_size_cols

        # Separate matrices
        date_matrix = data['timestampEvent'].values
        coint_price_matrix = data[coint_bid_price_cols].values
        lob_imbalance_matrix = data[lob_imbalance_cols].values
        bid_matrix = data[coint_bid_price_cols].values
        ask_matrix = data[coint_ask_price_cols].values
        order_book_matrix = pd.DataFrame(
            data[['timestampEvent'] + ask_price_cols + bid_price_cols + ask_size_cols + bid_size_cols]
        )

        # Precompute beta vectors and number of cointegrating relationships at each refresh point
        beta_vectors_list = []
        beta_indices = []
        beta_dates = []
        num_coints_list = []
        for idx in range(self.beta_refresh_frequency, N, self.beta_refresh_frequency):
            coin_basket_matrix = bid_matrix[idx-self.beta_refresh_frequency:idx]
            johansen_result = coint_johansen(coin_basket_matrix, det_order=0, k_ar_diff=1)
            trace_stats = johansen_result.lr1
            cv_95 = johansen_result.cvt[:, 1]
            num_coints = np.sum(trace_stats > cv_95)
            if num_coints == 0:
                # No cointegrating relationship found, mark as None if retest_cointegration == True
                if self.retest_cointegration:
                    beta_vectors_list.append(None)
                
                # If retest_cointegration == False, then we trade regardless
                else:
                    num_coints = 1
                    beta_vectors_list.append(johansen_result.evec[:, :num_coints])

            else:
                beta_vectors_list.append(johansen_result.evec[:, :num_coints])

            num_coints_list.append(num_coints)

            beta_indices.append(idx)
            beta_dates.append(date_matrix[idx])

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


        # Vectorized spread calculation (using most recen dt beta)
        spreads = np.full(N, np.nan)
        spread_means = np.full(N, np.nan)
        spread_stds = np.full(N, np.nan)
        z_scores = np.full(N, np.nan)
        normalized_betas = np.full((N, n_items), np.nan)
        notional_betas = np.full((N, n_items), np.nan)

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
                spreads[i] = coint_price_matrix[i] @ beta
                # Rolling window for mean/std
                if i >= self.spread_window:
                    window_mat = coint_price_matrix[i-self.spread_window:i+1]
                    window_spreads = window_mat @ beta
                    spread_means[i] = window_spreads.mean()
                    spread_stds[i] = window_spreads.std()
                    z_scores[i] = (spreads[i] - spread_means[i]) / spread_stds[i]

        # For each row, get execution price vector (with delay)
        exec_indices = np.clip(np.arange(N), 0, N-1)
        ask_exec = ask_matrix[exec_indices]
        bid_exec = bid_matrix[exec_indices]

        # For each row, choose price vector based on notional_beta sign
        price_vectors = np.where(notional_betas > 0, ask_exec, bid_exec)
        
       # Build history DataFrame
        history_dict = {
            'index': np.arange(N),
            'timestampEvent': date_matrix,
            'last_beta_vector_refresh': last_beta_refresh_date_per_row,
            'spread': spreads,
            'spread_mean': spread_means,
            'spread_std': spread_stds,
            'z_score': z_scores,
            'num_coints': num_coints_per_row,
        }

        # Add each order book column to history_df    
        for col in order_book_cols:
            history_dict[col] = order_book_matrix[col].values

        history_df = pd.DataFrame(history_dict)

        # --- Vectorized Trading Logic ---
        position = np.zeros(n_items)
        long = False
        short = False
        cash = cash_start
        cash_vec = [cash]
        position_history = [position.copy()]
        trade_log = []

        # Track ticks between enters and exits
        ticks_since_entry = None  # None means not in a position
        extends_since_entry = None  # None means not in a position
        last_beta_vector = None

        # 1, 2, 3
        # beta_vec = [0.4, -0.4, 0.1]

        # if z_score < -2:
        #   enter long
        #   
        # elif z_score > 2:
        #   enter short
        # -1 * beta_vec


        # exit
        # if z_score < -0.1:
        #   exit



        # Iterate through each data point
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
            if self.use_lob:
                # Get imbalance for each security
                imb_vec = lob_imbalance_matrix[i]

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
            if long and (z > self.enter_zscore + self.stop_loss_delta or z > -self.exit_zscore or beta_refresh):
                ticks_since_entry += 1
                before_position = position.copy()
                before_cash = cash
                before_position_value = position @ price_vector
                
                # Process trades for each security
                total_cash = 0
                new_position = np.zeros(n_items)
                level_fills_by_security = []
                for j, (exchange, security) in enumerate(self.basket):
                    if position[j] != 0:  # Only trade non-zero positions
                        # If position is positive, sell to close. If negative, buy to close
                        action = 'sell' if position[j] > 0 else 'buy'
                        trade_result = self.process_trade(
                            lob_data=order_book_matrix.iloc[i:i+1],
                            action=action,
                            order_type='market', 
                            size=abs(position[j]),
                            exchange=exchange,
                            security=security
                        )
                        if trade_result['filled']:
                            total_cash += trade_result['cash']
                            level_fills_by_security.append({
                                'exchange': exchange,
                                'security': security,
                                'fills': trade_result['level_fills']
                            })
                        else:
                            print(f"Order not filled: {trade_result}")
                
                cash = cash + total_cash
                after_cash = cash
                position = new_position
                after_position = position.copy()
                after_position_value = position @ price_vector
                long = False
                
                # Determine exit reason
                if z > self.enter_zscore + self.stop_loss_delta:
                    action = 'stop_loss_exit_long'
                elif beta_refresh:
                    action = 'beta_refresh_exit_long'
                else:
                    action = 'exit_long'
                
            # Exit short position (stop loss, normal exit, or beta refresh)
            elif short and (z < -self.enter_zscore - self.stop_loss_delta or z < self.exit_zscore or beta_refresh):
                ticks_since_entry += 1
                before_position = position.copy()
                before_cash = cash
                before_position_value = position @ price_vector

                # Process trades for each security
                total_cash = 0
                new_position = np.zeros(n_items)
                level_fills_by_security = []
                for j, (exchange, security) in enumerate(self.basket):
                    if position[j] != 0:  # Only trade non-zero positions
                        # If position is positive, sell to close. If negative, buy to close
                        action = 'sell' if position[j] > 0 else 'buy'
                        trade_result = self.process_trade(
                            lob_data=order_book_matrix.iloc[i:i+1],
                            action=action,
                            order_type='market',
                            size=abs(position[j]),
                            exchange=exchange,
                            security=security
                        )
                        if trade_result['filled']:
                            total_cash += trade_result['cash']
                            level_fills_by_security.append({
                                'exchange': exchange,
                                'security': security,
                                'fills': trade_result['level_fills']
                            })
                        else:
                            print(f"Order not filled: {trade_result}")

                cash = cash + total_cash
                after_cash = cash
                position = new_position
                after_position = position.copy()
                after_position_value = position @ price_vector
                short = False
                
                # Determine exit reason
                if z < -self.enter_zscore - self.stop_loss_delta:
                    action = 'stop_loss_exit_short'
                elif beta_refresh:
                    action = 'beta_refresh_exit_short'
                else:
                    action = 'exit_short'
                    
            # Enter/extend long (but not if stop loss would be triggered)
            elif z < -self.enter_zscore and (not self.use_lob or (self.use_lob and lob_signal)):
                # Only extend/enter if not past stop loss threshold
                if z > self.enter_zscore + self.stop_loss_delta:
                    # Do not extend/enter, treat as no trade (should not happen, but for safety)
                    if (long or short) and ticks_since_entry is not None:
                        ticks_since_entry += 1
                else:
                    # Skip if already in position and extends not allowed
                    if not (long and not self.use_extends):  # Changed to allow continuing execution
                        before_position = position.copy()
                        before_cash = cash
                        before_position_value = position @ price_vector
                        
                        # Scale notional based on z-score magnitude if dynamic sizing enabled
                        if self.use_dynamic_sizing:
                            z_scale = min(abs(z / self.enter_zscore), 3.0)  # Cap at 3x
                            scaled_notional_beta = notional_beta * z_scale
                            scaled_notional = notional * z_scale
                        else:
                            scaled_notional_beta = notional_beta
                            
                        delta_position = scaled_notional_beta / price_vector
                        
                        # Process trades for each security
                        total_cash = 0
                        new_position = position.copy()
                        level_fills_by_security = []
                        for j, (exchange, security) in enumerate(self.basket):
                            if delta_position[j] != 0:  # Only trade non-zero positions
                                trade_result = self.process_trade(
                                    lob_data=order_book_matrix.iloc[i:i+1],
                                    action='buy' if delta_position[j] > 0 else 'sell',
                                    order_type='market',
                                    size=abs(delta_position[j]),
                                    exchange=exchange,
                                    security=security
                                )
                                if trade_result['filled']:
                                    total_cash += trade_result['cash']
                                    new_position[j] += trade_result['size'] * (1 if delta_position[j] > 0 else -1)
                                    level_fills_by_security.append({
                                        'exchange': exchange,
                                        'security': security,
                                        'fills': trade_result['level_fills']
                                    })
                                else:
                                    print(f"Order not filled: {trade_result}")
                  
                        position = new_position
                        cash = cash + total_cash
                        after_cash = cash
                        after_position = position.copy()
                        after_position_value = position @ price_vector
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
            elif z > self.enter_zscore and (not self.use_lob or (self.use_lob and lob_signal)):
                # Only extend/enter if not past stop loss threshold
                if z < -self.enter_zscore - self.stop_loss_delta:
                    # Do not extend/enter, treat as no trade (should not happen, but for safety)
                    if (long or short) and ticks_since_entry is not None:
                        ticks_since_entry += 1
                else:
                    # Skip if already in position and extends not allowed
                    if not (short and not self.use_extends):  # Changed to allow continuing execution
                        before_position = position.copy()
                        before_cash = cash
                        before_position_value = position @ price_vector
                        
                        # Scale notional based on z-score magnitude if dynamic sizing enabled
                        if self.use_dynamic_sizing:
                            z_scale = min(abs(z / self.enter_zscore), 3.0)  # Cap at 3x
                            scaled_notional_beta = -notional_beta * z_scale
                            scaled_notional = notional * z_scale
                        else:
                            scaled_notional_beta = -notional_beta
                            
                        delta_position = scaled_notional_beta / price_vector
                        
                        # Process trades for each security
                        total_cash = 0
                        new_position = position.copy()
                        level_fills_by_security = []
                        for j, (exchange, security) in enumerate(self.basket):
                            if delta_position[j] != 0:  # Only trade non-zero positions
                                trade_result = self.process_trade(
                                    lob_data=order_book_matrix.iloc[i:i+1],
                                    action='buy' if delta_position[j] > 0 else 'sell',
                                    order_type='market',
                                    size=abs(delta_position[j]),
                                    exchange=exchange,
                                    security=security
                                )
                                if trade_result['filled']:
                                    total_cash += trade_result['cash']
                                    new_position[j] += trade_result['size'] * (1 if delta_position[j] > 0 else -1)
                                    level_fills_by_security.append({
                                        'exchange': exchange,
                                        'security': security,
                                        'fills': trade_result['level_fills']
                                    })
                                else:
                                    print(f"Order not filled: {trade_result}")
                        
                        position = new_position
                        cash = cash + total_cash
                        after_cash = cash
                        after_position = position.copy()
                        after_position_value = position @ price_vector
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
                    'timestampEvent': date_matrix[i],
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
                    'scaled_notional': scaled_notional if self.use_dynamic_sizing else notional,
                    'level_fills': level_fills_by_security
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
