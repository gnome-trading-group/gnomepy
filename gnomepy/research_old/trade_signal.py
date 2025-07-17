from gnomepy.data.types import Listing, Action, SignalType

class TradeSignal:
    """A class representing a trading signal for a security listing.
    
    Attributes:
        listing (Listing): The security listing the signal is for
        action (Action): The trading action - BUY, SELL, or NEUTRAL
        confidence (float): Confidence multiplier >= 1.0 indicating signal strength
        strategy (Strategy): The strategy that generated this signal
        signal_type (SignalType): The type of signal being generated
    """

    def __init__(self, listing: Listing, action: Action, confidence: float):
        self.listing = listing
        self.action = action
        self.confidence = confidence
    
    def __post_init__(self):
        """Validate confidence is >= 1.0."""
        if self.confidence < 1.0:
            raise ValueError("Confidence multiplier must be >= 1.0")
        
class BasketTradeSignal:
    """A class representing a list of signals for strategies that must trade baskets at specified proportions.
    
    Attributes:
        signals (list[TradeSignal]): The list of signals 
        proportions (list[float]): The list of proportions to trade each signal
    """
    def __init__(self, signals: list[TradeSignal], proportions: list[float], signal_type: SignalType, strategy):
        self.signals = signals
        self.proportions = proportions
        self.strategy = strategy
        self.signal_type = signal_type
