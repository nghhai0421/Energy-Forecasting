"""A fold container for the future deep-learning pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
