# Import the main functions you want exposed at the package level
from .simulation import generate_amplitude_modulated_pulses

# Optional: define what is exported when using `from sigint_examples import *`
__all__ = [
    "generate_amplitude_modulated_pulses",
]