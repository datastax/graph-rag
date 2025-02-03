from graph_retriever.utils.batched import batched


def test_batched():
    assert tuple(batched([], 2)) == ()
    assert tuple(batched([0, 1, 2, 3, 4], 2)) == ((0, 1), (2, 3), (4,))
    assert tuple(batched([0, 1, 2, 3, 4, 5], 2)) == ((0, 1), (2, 3), (4, 5))
