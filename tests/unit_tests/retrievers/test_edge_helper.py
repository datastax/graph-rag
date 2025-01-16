import pytest

from graph_pancake.retrievers.edge import Edge
from graph_pancake.retrievers.edge_helper import EdgeHelper


def test_edge_helper_initialization():
    edge_helper = EdgeHelper(["a", ("b", "c"), "b"])
    assert edge_helper.edges == [("a", "a"), ("b", "c"), ("b", "b")]


def test_get_incoming_outgoing_normalized():
    edge_helper = EdgeHelper([("href", "url")], use_normalized_metadata=True)
    assert edge_helper.get_incoming_outgoing({"href": "a", "url": "b"}) == (
        {Edge("url", "b")},
        {Edge("url", "a")},
    )

    assert edge_helper.get_incoming_outgoing({"href": ["a", "c"], "url": "b"}) == (
        {Edge("url", "b")},
        {Edge("url", "a"), Edge("url", "c")},
    )

    assert edge_helper.get_incoming_outgoing(
        {"href": ["a", "c"], "url": ["b", "d"]}
    ) == ({Edge("url", "b"), Edge("url", "d")}, {Edge("url", "a"), Edge("url", "c")})


def test_get_incoming_outgoing_denormalized():
    edge_helper = EdgeHelper(
        [("href", "url")],
        use_normalized_metadata=False,
        # Use non-default values so we can verify the fields are used.
        denormalized_path_delimiter=":",
        denormalized_static_value=57,
    )
    assert edge_helper.get_incoming_outgoing({"href": "a", "url": "b"}) == (
        {Edge("url", "b")},
        {Edge("url", "a")},
    )

    assert edge_helper.get_incoming_outgoing(
        {"href:a": 57, "href:c": 57, "url": "b"}
    ) == ({Edge("url", "b")}, {Edge("url", "a"), Edge("url", "c")})

    assert edge_helper.get_incoming_outgoing(
        {
            "href:a": 57,
            "href:c": 57,
            "url:b": 57,
            "url:d": 57,
        }
    ) == ({Edge("url", "b"), Edge("url", "d")}, {Edge("url", "a"), Edge("url", "c")})


def test_get_incoming_outgoing_unsupported_values():
    edge_helper = EdgeHelper(
        [("href", "url")],
        use_normalized_metadata=True,
    )

    # Unsupported value
    with pytest.raises(ValueError, match=r"Unsupported value .* in 'href'"):
        edge_helper.get_incoming_outgoing({"href": None})

    # Unsupported item value
    with pytest.raises(ValueError, match=r"Unsupported item value .* in 'href'"):
        edge_helper.get_incoming_outgoing({"href": [None]})

    edge_helper = EdgeHelper(
        [("href", "url")],
        use_normalized_metadata=False,
        # Use non-default values so we can verify the fields are used.
        denormalized_path_delimiter=":",
        denormalized_static_value=57,
    )
    # Unsupported value
    with pytest.raises(ValueError, match=r"Unsupported value .* in 'href'"):
        edge_helper.get_incoming_outgoing({"href": None})

    # It is OK for the list to exist in the metadata, although we do issue a warning
    # for that case.
    with pytest.warns(UserWarning, match="Normalized value [[]'a', 'c'] in 'href'"):
        assert edge_helper.get_incoming_outgoing(
            {
                "href": ["a", "c"],
            }
        ) == (set(), set())


def test_get_metadata_filter_normalized():
    edge_helper = EdgeHelper([], use_normalized_metadata=True)

    assert edge_helper.get_metadata_filter(edge=Edge(key="boolean", value=True)) == {
        "boolean": True
    }

    assert edge_helper.get_metadata_filter(edge=Edge(key="incoming", value=4)) == {
        "incoming": 4
    }

    assert edge_helper.get_metadata_filter(edge=Edge(key="place", value="berlin")) == {
        "place": "berlin"
    }


def test_get_metadata_filter_denormalized() -> None:
    edge_helper = EdgeHelper([], use_normalized_metadata=False)

    assert edge_helper.get_metadata_filter(edge=Edge(key="boolean", value=True)) == {
        "boolean": True
    }

    assert edge_helper.get_metadata_filter(
        edge=Edge(key="incoming", value=4), denormalize_edge=True
    ) == {"incoming.4": "$"}

    assert edge_helper.get_metadata_filter(
        edge=Edge(key="place", value="berlin"), denormalize_edge=True
    ) == {"place.berlin": "$"}
