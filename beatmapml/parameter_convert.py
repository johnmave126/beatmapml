from slider.mod import circle_radius

__all__ = [
    'calc_cs_propotion'
]


def calc_cs_propotion(cs: float) -> float:
    """Calculate the relative radius of a circle for a given circle size

        Args:
            cs (float): The circle size raw value.

        Returns:
            The relative radius of a circle. A float between 0 and 1,
            meaning the ratio between the raius and the playfield width.
    """
    return circle_radius(cs) / 512
