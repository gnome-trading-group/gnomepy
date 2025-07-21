from abc import ABC, abstractmethod
from gnomepy.data.types import SchemaBase, Intent, BasketIntent, Listing, SchemaType

# Global significance level mapping
SIGNIFICANCE_LEVEL_MAP = {0.01: 0, 0.05: 1, 0.10: 2}

class Signal(ABC):

    def __init__(self):
        self._id = id(self)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return isinstance(other, Signal) and self._id == other._id

    @abstractmethod
    def process_new_tick(self, data: SchemaBase) -> tuple[list[Intent], int]:
        """
        Process market data and return intents with processing latency.
        
        Returns:
            tuple: (list of Intent objects, elapsed time/latency in nanoseconds)
        """
        raise NotImplementedError


class PositionAwareSignal(Signal):

    @abstractmethod
    def process_new_tick(self, data: SchemaBase, positions: dict[Listing, int] = None) -> tuple[list[Intent], int]:
        """
        Process market data and return intents with processing latency, considering current positions.
        
        Args:
            data: Market data to process
            positions: Dictionary mapping Listing to current position size
        
        Returns:
            tuple: (list of Intent objects, elapsed time/latency in nanoseconds)
        """
        raise NotImplementedError


class CointegrationSignal(PositionAwareSignal):

    def __init__(self, listings: list[Listing], data_schema_type: SchemaType = SchemaType.MBP_10,
                 trade_frequency: int = 1, beta_refresh_frequency: int = 1000,
                 spread_window: int = 100, enter_zscore: float = 2.0, exit_zscore: float = 0.3,
                 stop_loss_delta: float = 0.0, retest_cointegration: bool = True, use_extends: bool = False,
                 use_lob: bool = False, use_dynamic_sizing: bool = True, significance_level: float = 0.05):
        """Initialize a cointegration trading strategy.
        
        Parameters
        ----------
        listings : list[Listing]
            List of listings to trade as a cointegrated basket
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
        significance_level : float, default 0.05
            Significance level for cointegration testing (0.01, 0.05, or 0.10)
        """

        # Get all the signal settings
        self.listings = listings
        self.data_schema_type = data_schema_type
        self.trade_frequency = trade_frequency
        self.beta_refresh_frequency = beta_refresh_frequency
        self.spread_window = spread_window
        self.enter_zscore = enter_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_delta = stop_loss_delta
        self.retest_cointegration = retest_cointegration
        self.use_extends = use_extends
        self.use_lob = use_lob
        self.use_dynamic_sizing = use_dynamic_sizing
        self.significance_level = significance_level

        # Prepare the signal to start generating intents
        self.initialize_signal()

    def initialize_signal(self):
        self.beta_vec = None
        self.norm_beta_vec = None
        self.n_coints = None
        self.beta_history = []
        self.beta_timestamps = []
        self.elapsed_ticks = 0
        return

    def process_new_tick(self, data: SchemaBase, positions: dict[Listing, float] = None) -> tuple[list[BasketIntent], int]:

        return
