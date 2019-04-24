
from typing import Tuple

class BoundingBox:
    """Bounding box representation:
       - coordinates: (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
       - confidence: a float value ranging between 0.0 and 1.0 that represents the probability that a face exists within these coordinates
    """
    def __init__(self, coordinates: Tuple[int, int, int, int], confidence: float):
        self.__coordinates: Tuple[int, int, int, int] = coordinates
        self.__confidence: float = confidence
    
    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        return self.__coordinates
    
    @property
    def confidence(self) -> float:
        return self.__confidence