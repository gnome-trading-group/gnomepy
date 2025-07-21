from abc import ABC, abstractmethod
from typing import List, Tuple
import random


class QueueModel(ABC):
    """Abstract base class for queue position models"""
    
    @abstractmethod
    def get_queue_position(self, order_size: int, price_level_size: int) -> int:
        """
        Get the queue position for an order at a given price level.
        
        Args:
            order_size: Size of the order
            price_level_size: Total size at the price level
            
        Returns:
            Queue position (0 = front of queue, higher = further back)
        """
        raise NotImplementedError
    
    @abstractmethod
    def should_execute_at_position(self, queue_position: int, execution_probability: float) -> bool:
        """
        Determine if an order should execute at a given queue position.
        
        Args:
            queue_position: Position in the queue
            execution_probability: Base probability of execution
            
        Returns:
            True if order should execute, False otherwise
        """
        raise NotImplementedError


class SimpleQueueModel(QueueModel):
    """Simple queue model that implements basic FIFO behavior"""
    
    def __init__(self, base_execution_probability: float = 0.8):
        self.base_execution_probability = base_execution_probability
    
    def get_queue_position(self, order_size: int, price_level_size: int) -> int:
        """Simple FIFO queue position - orders are added to the end"""
        return price_level_size
    
    def should_execute_at_position(self, queue_position: int, execution_probability: float) -> bool:
        """Simple execution model - front of queue has higher probability"""
        if queue_position == 0:
            return random.random() < execution_probability
        
        # Decay probability based on queue position
        decay_factor = max(0.1, 1.0 - (queue_position * 0.1))
        adjusted_probability = execution_probability * decay_factor
        
        return random.random() < adjusted_probability


class RealisticQueueModel(QueueModel):
    """More realistic queue model that considers order size and market conditions"""
    
    def __init__(self, base_execution_probability: float = 0.8, size_advantage_factor: float = 0.2):
        self.base_execution_probability = base_execution_probability
        self.size_advantage_factor = size_advantage_factor
    
    def get_queue_position(self, order_size: int, price_level_size: int) -> int:
        """
        Calculate queue position considering order size advantage.
        Larger orders may get priority in some cases.
        """
        # Simple model: larger orders get slight priority
        if order_size > price_level_size * 0.1:  # Order is >10% of level size
            # Large orders get some priority
            priority_bonus = int(order_size / price_level_size * 10)
            return max(0, price_level_size - priority_bonus)
        else:
            return price_level_size
    
    def should_execute_at_position(self, queue_position: int, execution_probability: float) -> bool:
        """
        Realistic execution model with queue position decay and randomness.
        """
        if queue_position == 0:
            return random.random() < execution_probability
        
        # Exponential decay based on queue position
        decay_factor = max(0.05, (0.9 ** queue_position))
        adjusted_probability = execution_probability * decay_factor
        
        # Add some randomness to simulate market noise
        noise = random.uniform(-0.1, 0.1)
        final_probability = max(0.0, min(1.0, adjusted_probability + noise))
        
        return random.random() < final_probability