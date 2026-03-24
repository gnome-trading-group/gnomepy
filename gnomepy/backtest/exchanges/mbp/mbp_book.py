import bisect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from gnomepy.backtest.exchanges.mbp.types import LocalOrder
from gnomepy.backtest.queues.base import QueueModel
from gnomepy.data.types import BidAskPair, Order, OrderType, MBP10, MBP1


@dataclass
class OrderBookLevel:
    price: int
    size: int

    local_orders: Deque[LocalOrder] = field(default_factory=deque)

    @property
    def has_local_orders(self):
        return bool(len(self.local_orders))

@dataclass
class OrderMatch:
    price: int
    size: int

@dataclass
class MBPBook:
    queue_model: QueueModel

    bids: list[int] = field(default_factory=list)  # descending prices
    asks: list[int] = field(default_factory=list)  # ascending prices
    price_to_level: dict[str, dict[int, OrderBookLevel]] = field(default_factory=lambda: {'B': {}, 'A': {}})

    local_orders: dict[str, dict[str, LocalOrder]] = field(default_factory=lambda: {'B': {}, 'A': {}})

    def get_best_bid(self) -> int | None:
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> int | None:
        return self.asks[0] if self.asks else None

    def _insert_price(self, price: int, side: str):
        key = (lambda x: -x) if side == 'B' else None
        bisect.insort(self.bids if side == 'B' else self.asks, price, key=key)

    def on_market_update(self, levels: list[BidAskPair]) -> list[tuple[LocalOrder, int]]:
        curr = {'B': {}, 'A': {}}
        for lvl in levels:
            curr['B'][lvl.bid_px] = lvl.bid_sz
            curr['A'][lvl.ask_px] = lvl.ask_sz

        for side in ['B', 'A']:
            prev = self.price_to_level[side]
            for price in set(prev) | set(curr[side]):
                prev_level = prev.get(price, OrderBookLevel(price=price, size=0))
                new_size = curr[side].get(price, 0)
                lst = self.bids if side == 'B' else self.asks
                if new_size == 0:
                    if not prev_level.has_local_orders:
                        self.price_to_level[side].pop(price)
                        lst.remove(price)
                        continue

                if price not in lst:
                    self._insert_price(price, side)

                self.queue_model.on_modify(prev_level.size, new_size, prev_level.local_orders)

                prev_level.size = new_size
                prev[price] = prev_level

        if not self.local_orders:
            return []

        # Check if any of the changes have filled our local orders
        # This happens when the book crosses (best_bid >= best_ask)
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if not best_bid or not best_ask or best_bid < best_ask:
            return []
        
        # Book is crossed - need to check for fills
        all_fills = []
        
        # Check local bid orders - they can be filled if bid_price >= best_ask
        # We need to check bids in descending order (best first)
        for bid_price in sorted(self.bids, reverse=True):
            if bid_price < best_ask:
                break  # No more bids that can be filled
            
            # Check if we have local bid orders at this price
            bid_level = self.price_to_level['B'].get(bid_price)
            if not bid_level or not bid_level.has_local_orders:
                continue
            
            # This bid can be filled against the ask side
            # We need to find matching ask levels
            remaining_to_fill = sum(loc.remaining for loc in bid_level.local_orders)
            
            # Match against ask levels starting from best_ask
            for ask_price in self.asks:
                if ask_price > bid_price:
                    break  # Ask price too high for this bid
                
                if remaining_to_fill == 0:
                    break
                
                ask_level = self.price_to_level['A'].get(ask_price)
                if not ask_level:
                    continue
                
                # Use queue model to determine fills
                available_size = ask_level.size
                if available_size == 0:
                    continue
                
                # Process fills using queue model
                trade_size = min(remaining_to_fill, available_size)
                fills = self.queue_model.on_trade(trade_size, bid_level.local_orders)
                all_fills.extend(fills)
                
                # Update remaining to fill
                filled_qty = sum(fill[1] for fill in fills)
                remaining_to_fill -= filled_qty
                
                # Update ask level size (consumed by our fills)
                ask_level.size -= filled_qty
        
        # Check local ask orders - they can be filled if ask_price <= best_bid
        # We need to check asks in ascending order (best first)
        for ask_price in sorted(self.asks):
            if ask_price > best_bid:
                break  # No more asks that can be filled
            
            # Check if we have local ask orders at this price
            ask_level = self.price_to_level['A'].get(ask_price)
            if not ask_level or not ask_level.has_local_orders:
                continue
            
            # This ask can be filled against the bid side
            # We need to find matching bid levels
            remaining_to_fill = sum(loc.remaining for loc in ask_level.local_orders)
            
            # Match against bid levels starting from best_bid
            for bid_price in sorted(self.bids, reverse=True):
                if bid_price < ask_price:
                    break  # Bid price too low for this ask
                
                if remaining_to_fill == 0:
                    break
                
                bid_level = self.price_to_level['B'].get(bid_price)
                if not bid_level:
                    continue
                
                # Use queue model to determine fills
                available_size = bid_level.size
                if available_size == 0:
                    continue
                
                # Process fills using queue model
                trade_size = min(remaining_to_fill, available_size)
                fills = self.queue_model.on_trade(trade_size, ask_level.local_orders)
                all_fills.extend(fills)
                
                # Update remaining to fill
                filled_qty = sum(fill[1] for fill in fills)
                remaining_to_fill -= filled_qty
                
                # Update bid level size (consumed by our fills)
                bid_level.size -= filled_qty
        
        # Clear filled orders
        self._clear_fills(all_fills)
        
        return all_fills

    def on_trade(self, order: MBP10 | MBP1) -> list[tuple[LocalOrder, int]]:
        if not self.local_orders:
            return []

        opp = 'A' if order.side == 'B' else 'B'
        all_fills = []

        remaining_size = order.size
        opp_idx = 0
        opp_list = self.bids if opp == 'B' else self.asks
        while remaining_size > 0 and opp_idx < len(opp_list):
            price = opp_list[opp_idx]
            if (price < order.price and opp == 'B') or (price > order.price and opp == 'A'):
                break

            level = self.price_to_level[opp].get(price)
            if not level:
                raise ValueError("Malformed local book")

            fills = self.queue_model.on_trade(remaining_size, level.local_orders)
            all_fills.extend(fills)

            remaining_size -= sum([fill[1] for fill in fills])

            left_to_consume = min(remaining_size, level.size)
            remaining_size -= left_to_consume
            level.size -= left_to_consume

            opp_idx += 1

        self._clear_fills(all_fills)
        return all_fills

    def _clear_fills(self, fills: list[tuple[LocalOrder, int]]):
        for (local_order, _) in fills:
            if local_order.remaining == 0:
                del self.local_orders[local_order.order.side][local_order.order.client_oid]

    def add_local_order(self, order: Order, remaining: int | None = None) -> None:
        side, price = order.side, order.price
        order.client_oid = order.client_oid or f"internal_{time.time_ns()}"

        if order.client_oid in self.local_orders[side]:
            raise ValueError(f"Duplicate client OID: {order.client_oid}")

        current = self.price_to_level[side].get(price)
        if current is None: # new level in book
            current = OrderBookLevel(price=price, size=0)
            self._insert_price(price, side)
        remaining = remaining or order.size
        loc = LocalOrder(order=order, remaining=remaining, phantom_volume=current.size)
        self.local_orders[side][order.client_oid] = loc

        current.local_orders.append(loc)
        self.price_to_level[side][price] = current

    def cancel_order(self, client_oid: str) -> bool:
        side = None
        for s in ['B', 'A']:
            if client_oid in self.local_orders[s]:
                side = s
                break
        if not side:
            return False

        loc = self.local_orders[side].pop(client_oid)
        dq = self.price_to_level[side][loc.order.price].local_orders
        if loc in dq:
            dq.remove(loc)
            return True
        return False

    def get_matching_orders(self, order: Order) -> list[OrderMatch]:
        matches = []

        remaining_size = order.size
        comparison = lambda _price: (_price > order.price) if order.side == "B" else (_price < order.price)
        target_side, target_prices = ("A", self.asks) if order.side == "B" else ("B", self.bids)

        for price in target_prices:
            if order.order_type == OrderType.LIMIT and comparison(price):
                break

            if remaining_size == 0:
                break

            price_level = self.price_to_level[target_side][price]
            
            # Skip price levels that have our own local orders to avoid self-trading
            # Our local orders are queued behind the market depth, but to be safe,
            # we skip matching against any price level where we have orders
            if price_level.has_local_orders:
                raise ValueError(f"Self filling triggered - price level {price} has local orders")
                # continue

            if price_level.size == 0:
                raise ValueError(f"Malformed local book - price level has no size but no local orders: {price_level}")
                # continue

            match_size = min(remaining_size, price_level.size)
            remaining_size -= match_size
            matches.append(OrderMatch(price=price, size=match_size))

        return matches
