# growth_signal_bot.py - backwards compatibility shim
# The canonical implementation has been renamed to LongSignalStrategy.
from .long_signal_strategy import LongSignalStrategy as GrowthSignalBot

__all__ = ["GrowthSignalBot"]
