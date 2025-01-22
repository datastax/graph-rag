import pytest
from langchain_graph_retriever.strategies import (
    Eager,
    Mmr,
)
from langchain_graph_retriever.strategies.utils import build_strategy


def test_build_strategy_base():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with no changes
    strategy = build_strategy(base_strategy=base_strategy)
    assert strategy == base_strategy

    # base strategy with base_k override
    strategy = build_strategy(base_strategy=base_strategy, base_k=9)
    assert strategy == Eager(k=9, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with changed k
    strategy = build_strategy(base_strategy=base_strategy, k=7)
    assert strategy == Eager(k=7, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with base_k and changed k
    strategy = build_strategy(base_strategy=base_strategy, base_k=9, k=7)
    assert strategy == Eager(k=7, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with invalid kwarg
    with pytest.warns(UserWarning, match="Unsupported key 'invalid_kwarg' set"):
        strategy = build_strategy(base_strategy=base_strategy, invalid_kwarg=4)
        assert strategy == base_strategy


def test_build_strategy_base_override():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Eager(k=7, start_k=4, adjacent_k=8, max_depth=3)

    # override base strategy
    strategy = build_strategy(
        base_strategy=base_strategy, k=4, strategy=override_strategy
    )
    assert strategy == override_strategy

    # override base strategy and change params
    strategy = build_strategy(
        base_strategy=base_strategy, strategy=override_strategy, k=3, adjacent_k=7
    )
    assert strategy == Eager(k=3, start_k=4, adjacent_k=7, max_depth=3)

    # override base strategy and invalid kwarg
    with pytest.warns(UserWarning, match="Unsupported key 'invalid_kwarg' set"):
        strategy = build_strategy(
            base_strategy=base_strategy,
            k=4,
            strategy=override_strategy,
            invalid_kwarg=4,
        )
        assert strategy == override_strategy

    # override base strategy with dict and change params
    strategy = build_strategy(
        base_strategy=base_strategy,
        strategy={"k": 9, "start_k": 7, "adjacent_k": 11},
        adjacent_k=7,
    )
    assert strategy == Eager(k=9, start_k=7, adjacent_k=7, max_depth=2)

    # override base strategy with dict (with invalid kwarg) and change params
    with pytest.warns(
        UserWarning, match="Unsupported key 'invalid_kwarg' in 'strategy' set, ignored."
    ):
        strategy = build_strategy(
            base_strategy=base_strategy,
            strategy={"k": 9, "start_k": 7, "invalid_kwarg": 11},
            adjacent_k=7,
        )
        assert strategy == Eager(k=9, start_k=7, adjacent_k=7, max_depth=2)


def test_build_strategy_base_override_mmr():
    base_strategy = Eager(k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Mmr(k=7, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.3)

    # override base strategy with mmr kwarg
    with pytest.warns(UserWarning, match="Unsupported key 'lambda_mult' set, ignored."):
        strategy = build_strategy(base_strategy=base_strategy, lambda_mult=0.2)
        assert strategy == base_strategy

    # override base strategy with mmr strategy
    strategy = build_strategy(
        base_strategy=base_strategy, k=4, strategy=override_strategy
    )
    assert strategy == override_strategy

    # override base strategy with mmr strategy and mmr arg
    strategy = build_strategy(
        base_strategy=base_strategy, k=4, strategy=override_strategy, lambda_mult=0.2
    )
    assert strategy == Mmr(k=7, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.2)

    # start with override strategy, change to base, try to set mmr arg
    with pytest.warns(UserWarning, match="Unsupported key 'lambda_mult' set, ignored."):
        build_strategy(
            base_strategy=override_strategy, strategy=base_strategy, lambda_mult=0.2
        )


def test_build_strategy_base_none():
    override_strategy = Eager(k=7, start_k=4, adjacent_k=8, max_depth=3)

    # no base strategy, attempt to set args
    with pytest.raises(
        ValueError, match="Error: 'strategy' must be set before other args"
    ):
        build_strategy(k=6, start_k=5, adjacent_k=9, max_depth=2)

    # no base strategy, attempt to set strategy via dict
    with pytest.raises(ValueError, match="Unsupported 'strategy'"):
        build_strategy(
            strategy={"k": 6, "start_k": 5, "adjacent_k": 9, "max_depth": 2},
            k=6,
            start_k=5,
            adjacent_k=9,
            max_depth=2,
        )

    # no base strategy, set strategy with override
    strategy = build_strategy(strategy=override_strategy)
    assert strategy == override_strategy

    # no base strategy, set strategy with override and other args
    strategy = build_strategy(strategy=override_strategy, k=6, start_k=5, adjacent_k=9)
    assert strategy == Eager(k=6, start_k=5, adjacent_k=9, max_depth=3)

    # no base strategy, set strategy with override and invalid args
    # base strategy with invalid kwarg
    with pytest.warns(UserWarning, match="Unsupported key 'invalid_kwarg' set"):
        strategy = build_strategy(strategy=override_strategy, invalid_kwarg=4)
        assert strategy == override_strategy
