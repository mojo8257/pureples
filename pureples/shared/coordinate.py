from typing import Optional, Tuple, List


class Coordinate:
    """
    Represents a point in 2D or 3D space with automatic normalization and z-axis defaulting.

    Attributes:
        x (float): x-coordinate in [-1, 1].
        y (float): y-coordinate in [-1, 1].
        z (float): z-coordinate in [-1, 1], defaults to 0 if not provided.
    """

    def __init__(self, x: float, y: float, z: Optional[float] = None):
        self.x = self._normalize(x)
        self.y = self._normalize(y)
        # If z is None, default to 0.0
        self.z = 0.0 if z is None else self._normalize(z)

    @staticmethod
    def _normalize(value: float) -> float:
        """
        Clamp and normalize the given value into the range [-1.0, 1.0].

        Args:
            value (float): The value to normalize.

        Returns:
            float: The normalized value.
        """
        return max(-1.0, min(1.0, value))

    def to_cppn_input(self, other: "Coordinate", outgoing: bool) -> List[float]:
        """
        Generate a CPPN input vector from this coordinate to another coordinate.

        Args:
            other (Coordinate): The destination coordinate.
            outgoing (bool): Direction flag for bias usage if needed.

        Returns:
            List[float]: A 7-dimensional input vector [x1, y1, z1, x2, y2, z2, bias].
        """
        bias = 1.0
        if outgoing:
            # source to destination
            return [
                self.x, self.y, self.z,
                other.x, other.y, other.z,
                bias
            ]
        else:
            # destination to source (reverse)
            return [
                other.x, other.y, other.z,
                self.x, self.y, self.z,
                bias
            ]

    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Return the coordinate as a tuple (x, y, z).

        Returns:
            Tuple[float, float, float]
        """
        return self.x, self.y, self.z

    def __eq__(self, other: object) -> bool:
        """
        Compare two coordinates for equality.

        Args:
            other (object): Another coordinate to compare.

        Returns:
            bool: True if coordinates are equal element-wise.
        """
        if not isinstance(other, Coordinate):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the coordinate.

        Returns:
            str: String in the form 'Coordinate(x, y, z)'.
        """
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __hash__(self):
        return hash((self.x, self.y, self.z))
