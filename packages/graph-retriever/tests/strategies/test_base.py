import dataclasses

import pytest
from graph_retriever.strategies import (
    Eager,
    Mmr,
    Strategy,
)


def test_build_strategy_base():
    base_strategy = Eager(select_k=6, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with no changes
    strategy = Strategy.build(base_strategy=base_strategy)
    assert strategy == base_strategy

    # base strategy with changed k
    strategy = Strategy.build(base_strategy=base_strategy, select_k=7)
    assert strategy == Eager(select_k=7, start_k=5, adjacent_k=9, max_depth=2)

    # base strategy with invalid kwarg
    with pytest.raises(
        TypeError, match=r"got an unexpected keyword argument 'invalid_kwarg'"
    ):
        strategy = Strategy.build(base_strategy=base_strategy, invalid_kwarg=4)
        assert strategy == base_strategy


def test_build_strategy_base_override():
    base_strategy = Eager(select_k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Eager(select_k=7, start_k=4, adjacent_k=8, max_depth=3)

    # override base strategy
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, select_k=4
    )
    assert strategy == dataclasses.replace(override_strategy, select_k=4)

    # override base strategy and change params
    strategy = Strategy.build(
        base_strategy=base_strategy,
        strategy=override_strategy,
        select_k=3,
        adjacent_k=7,
    )
    assert strategy == Eager(select_k=3, start_k=4, adjacent_k=7, max_depth=3)

    # override base strategy and invalid kwarg
    with pytest.raises(
        TypeError, match=r"got an unexpected keyword argument 'invalid_kwarg'"
    ):
        strategy = Strategy.build(
            base_strategy=base_strategy,
            strategy=override_strategy,
            select_k=4,
            invalid_kwarg=4,
        )

    # attempt override base strategy with dict
    with pytest.raises(ValueError, match="Unsupported 'strategy'"):
        strategy = Strategy.build(
            base_strategy=base_strategy,
            strategy={"k": 9, "start_k": 7, "adjacent_k": 11},
        )


def test_build_strategy_base_override_mmr():
    base_strategy = Eager(select_k=6, start_k=5, adjacent_k=9, max_depth=2)
    override_strategy = Mmr(
        select_k=7, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.3
    )

    # override base strategy with mmr kwarg
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'lambda_mult'"
    ):
        strategy = Strategy.build(base_strategy=base_strategy, lambda_mult=0.2)
        assert strategy == base_strategy

    # override base strategy with mmr strategy
    strategy = Strategy.build(
        base_strategy=base_strategy, strategy=override_strategy, select_k=4
    )
    assert strategy == dataclasses.replace(override_strategy, select_k=4)

    # override base strategy with mmr strategy and mmr arg
    strategy = Strategy.build(
        base_strategy=base_strategy,
        strategy=override_strategy,
        select_k=4,
        lambda_mult=0.2,
    )
    assert strategy == Mmr(
        select_k=4, start_k=4, adjacent_k=8, max_depth=3, lambda_mult=0.2
    )

    # start with override strategy, change to base, try to set mmr arg
    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'lambda_mult'"
    ):
        Strategy.build(
            base_strategy=override_strategy, strategy=base_strategy, lambda_mult=0.2
        )


def test_setting_k_sets_select_k():
    assert Eager(select_k=4) == Eager(k=4)
    assert Mmr(select_k=3) == Mmr(k=3)
