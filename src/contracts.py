"""A fold container for the future deep-learning pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Store immutable training and validation row indices.

    Parameters
    ----------
    train_indices : tuple of int
        Positions used for model fitting.
    validation_indices : tuple of int
        Positions used for validation.
    """

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
