import warnings
from typing import Any

from .base import Strategy


def build_strategy(
    base_k: int | None = None,
    base_strategy: Strategy | None = None,
    **invoke_kwargs: Any,
) -> Strategy:
    """Build a strategy for an retrieval.

    Build a strategy for an retrieval from the base strategy, any strategy passed in
    the invocation, and any related key word arguments.
    """
    strategy: Strategy | None = None
    if base_strategy is not None:
        # Deep copy in case the strategy has mutable state
        strategy = base_strategy.model_copy(deep=True)
        if base_k is not None:
            strategy.k = base_k
        for key, value in invoke_kwargs.items():
            if key == "strategy":
                if isinstance(value, Strategy):
                    strategy = value.model_copy(deep=True)
                elif isinstance(value, dict):
                    for k in value.keys():
                        if k not in strategy.model_fields.keys():
                            warnings.warn(
                                f"Unsupported key '{k}' in 'strategy' set, ignored."
                            )
                    strategy = strategy.model_copy(update=value)
                else:
                    raise ValueError(f"Unsupported strategy {value}")
            elif key in strategy.model_fields.keys():
                strategy = strategy.model_copy(update={key: value})
            else:
                warnings.warn(f"Unsupported key '{key}' set, ignored.")

    else:  # no base strategy
        for key, value in invoke_kwargs.items():
            if key == "strategy":
                if isinstance(value, Strategy):
                    strategy = value.model_copy(deep=True)
                else:
                    raise ValueError(f"Unsupported 'strategy': {value}")
            else:
                if strategy is None:
                    raise ValueError("Error: 'strategy' must be set before other args.")
                elif key in strategy.model_fields.keys():
                    strategy = strategy.model_copy(update={key: value})
                else:
                    warnings.warn(f"Unsupported key '{key}' set, ignored.")
    if strategy is None:
        raise ValueError("'strategy' must be set.")
    return strategy
