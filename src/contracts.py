"""A fold container for the future deep-learning pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """Store train and validation index tuples in an immutable fold."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
