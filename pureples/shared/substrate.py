"""
The substrate.
"""

"""3D_update"""
# 3D_update--------------------------------------------------------------------------------
from typing import List, Tuple
from pureples.shared.coordinate import Coordinate

class Substrate(object):
    """
    Represents a substrate: Input coordinates, output coordinates,
    hidden coordinates and a resolution defaulting to 10.0.
    """

    def __init__(self,
                 input_coordinates: List[Tuple[float, float]],
                 output_coordinates: List[Tuple[float, float]],
                 hidden_coordinates: List[Tuple[float, float]] = (),
                 res: float = 10.0):
        # Convert all coordinate tuples to Coordinate instances
        self.input_coordinates: List[Coordinate] = [Coordinate(x, y) for x, y in input_coordinates]
        self.hidden_coordinates: List[Coordinate] = [Coordinate(x, y) for x, y in hidden_coordinates]
        self.output_coordinates: List[Coordinate] = [Coordinate(x, y) for x, y in output_coordinates]
        self.res: float = res
# 3D_update--------------------------------------------------------------------------------