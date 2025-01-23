import pytest
from langchain_graph_retriever.strategies import (
    Eager,
    Mmr,
)
from langchain_graph_retriever.strategies.base import Strategy


def test_build_strategy_base():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with no changes
    strategy = Strategy.build(base_strategy=base_strategy)
    assert strategy == base_strategy

    # base strategy with base_k override
    strategy = Strategy.build(base_strategy=base_strategy, base_k=9)
    assert strategy == Eager(k=9, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with changed k
    strategy = Strategy.build(base_strategy=base_strategy, k=7)
    assert strategy == Eager(k=7, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with base_k and changed k
    strategy = Strategy.build(base_strategy=base_strategy, base_k=9, k=7)
    assert strategy == Eager(k=7, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with invalid kwarg
    with pytest.warns(UserWarning, match=r"Unsupported key\(s\) 'invalid_kwarg' set."):
        strategy = Strategy.build(base_strategy=base_strategy, invalid_kwarg=4)
        assert strategy == base_strategy


def test_build_strategy_base_override():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Eager(k=7, start_k=4, adjacent_k=8, max_depth=3)

    # override base strategy
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, k=4
    )
    assert strategy == override_strategy.model_copy(update={"k": 4})

    # override base strategy with base_k should prefer the override
    strategy = Strategy.build(
        base_strategy=base_strategy, base_k=4, strategy=override_strategy
    )
    assert strategy == override_strategy

    # override base strategy and change params
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, k=3, adjacent_k=7
    )
    assert strategy == Eager(k=3, start_k=4, adjacent_k=7, max_depth=3)

    # override base strategy and invalid kwarg
    with pytest.warns(UserWarning, match=r"Unsupported key\(s\) 'invalid_kwarg' set."):
        strategy = Strategy.build(
            base_strategy=base_strategy,
            strategy=override_strategy,
            k=4,
            invalid_kwarg=4,
        )
        assert strategy == override_strategy.model_copy(update={"k": 4})

    # attempt override base strategy with dict
    with pytest.raises(ValueError, match="Unsupported 'strategy'"):
        strategy = Strategy.build(
            base_strategy=base_strategy,
            strategy={"k": 9, "start_k": 7, "adjacent_k": 11},
        )



def test_build_strategy_base_override_mmr():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Mmr(k=7, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.3)

    # override base strategy with mmr kwarg
    with pytest.warns(UserWarning, match=r"Unsupported key\(s\) 'lambda_mult' set."):
        strategy = Strategy.build(base_strategy=base_strategy, lambda_mult=0.2)
        assert strategy == base_strategy

    # override base strategy with mmr strategy
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, k=4
    )
    assert strategy == override_strategy.model_copy(update={"k": 4})

    # override base strategy with mmr strategy and mmr arg
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, k=4, lambda_mult=0.2
    )
    assert strategy == Mmr(k=4, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.2)

    # start with override strategy, change to base, try to set mmr arg
    with pytest.warns(UserWarning, match=r"Unsupported key\(s\) 'lambda_mult' set."):
        Strategy.build(
            base_strategy=override_strategy, strategy=base_strategy, lambda_mult=0.2
        )


def test_build_strategy_base_none():
    override_strategy = Eager(k=7, start_k=4, adjacent_k=8, max_depth=3)

    # no base strategy, attempt to set args
    with pytest.raises(
        ValueError, match="'strategy' must be set in `__init__` or invocation"
    ):
        Strategy.build(k=6, start_k=5, adjacent_k=9, max_depth=2)

    # no base strategy, attempt to set strategy via dict
    with pytest.raises(ValueError, match="Unsupported 'strategy'"):
        Strategy.build(
            strategy={"k": 6, "start_k": 5, "adjacent_k": 9, "max_depth": 2},
            k=6,
            start_k=5,
            adjacent_k=9,
            max_depth=2,
        )

    # no base strategy, set strategy with override
    strategy = Strategy.build(strategy=override_strategy)
    assert strategy == override_strategy

    # no base strategy, set strategy with override and other args
    strategy = Strategy.build(strategy=override_strategy, k=6, start_k=5, adjacent_k=9)
    assert strategy == Eager(k=6, start_k=5, adjacent_k=9, max_depth=3)

    # no base strategy, set strategy with override and invalid args
    # base strategy with invalid kwarg
    with pytest.warns(UserWarning, match=r"Unsupported key\(s\) 'invalid_kwarg' set."):
        strategy = Strategy.build(strategy=override_strategy, invalid_kwarg=4)
        assert strategy == override_strategy
