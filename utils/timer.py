"""Timer Utilities for Negotiation Management"""

import time
import threading
from typing import Callable, Optional
import logging

class NegotiationTimer:
    """Manages timing for negotiations with callbacks"""
    
    def init(self, time_limit: int = 180):
        """
        Initialize timer
        
        Args:
            time_limit: Time limit in seconds
        """
        self.time_limit = time_limit
        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.timeout_callback = None
        self.timer_thread = None
        self.logger = logging.getLogger(name)
        
    def start(self, timeout_callback: Optional[Callable] = None):
        """Start the negotiation timer"""
        self.start_time = time.time()
        self.is_running = True
        self.timeout_callback = timeout_callback
        
        if timeout_callback:
            self.timer_thread = threading.Timer(self.time_limit, self._timeout_handler)
            self.timer_thread.start()
        
        self.logger.debug(f"Timer started with {self.time_limit}s limit")
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
        self.is_running = False
        
        if self.timer_thread:
            self.timer_thread.cancel()
        
        self.logger.debug("Timer stopped")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds"""
        elapsed = self.get_elapsed_time()
        return max(0, self.time_limit - elapsed)
    
    def is_timeout(self) -> bool:
        """Check if timer has exceeded limit"""
        return self.get_elapsed_time() >= self.time_limit
    
    def _timeout_handler(self):
        """Handle timeout event"""
        self.logger.info("Negotiation timeout reached")
        if self.timeout_callback and self.is_running:
            self.timeout_callback()
