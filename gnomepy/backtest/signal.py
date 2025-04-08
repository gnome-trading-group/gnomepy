from gnomepy.data.types import *

class Signal:
    def __init__(self, name: str, columns: list[str], pd_expression: str, np_expression: str):
        self.name = name
        self.columns = columns
        self.pd_expression = pd_expression
        self.np_expression = np_expression



global_signals = {
    "bid0_ma_30": Signal(
        name='bid0_ma_30',
        columns=['bid0'],
        pd_expression="df['bid0'].rolling(window=30).mean()",
        np_expression="np.convolve(bid0, np.ones(30)/30)"
    )
}
