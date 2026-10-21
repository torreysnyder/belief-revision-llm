import pytest
from belief_revision.extensions import (
    EXTENSION_REGISTRY,
    ConversationExtension,
    register_extension,
    resolve_extension_factories,
)
from collections.abc import Iterator


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    original = EXTENSION_REGISTRY.copy()
    EXTENSION_REGISTRY.clear()
    try:
        yield
    finally:
        EXTENSION_REGISTRY.clear()
        EXTENSION_REGISTRY.update(original)


def first_factory() -> ConversationExtension:
    return ConversationExtension()


def second_factory() -> ConversationExtension:
    return ConversationExtension()


def test_register_and_resolve_extensions_in_cli_order() -> None:
    register_extension("first", first_factory)
    register_extension("second", second_factory)

    assert resolve_extension_factories(("second", "first", "second")) == (
        second_factory,
        first_factory,
        second_factory,
    )


def test_resolve_applies_named_configuration() -> None:
    register_extension("first", first_factory)

    configured_factory = resolve_extension_factories(
        ("first",),
        configurations={"first": {"unused": "value"}},
    )[0]

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        configured_factory()


def test_empty_registry_and_empty_selection_are_valid() -> None:
    assert EXTENSION_REGISTRY == {}
    assert resolve_extension_factories(()) == ()


@pytest.mark.parametrize("name", ["", None, 12])
def test_register_rejects_invalid_names(name: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        register_extension(name, first_factory)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", [" extension", "extension ", " extension "])
def test_register_rejects_surrounding_whitespace(name: str) -> None:
    with pytest.raises(ValueError, match="surrounding whitespace"):
        register_extension(name, first_factory)


def test_register_rejects_non_callable_factory() -> None:
    with pytest.raises(TypeError, match="factory must be callable"):
        register_extension("broken", object())  # type: ignore[arg-type]


def test_register_rejects_duplicate_name() -> None:
    register_extension("duplicate", first_factory)

    with pytest.raises(ValueError, match="already registered: 'duplicate'"):
        register_extension("duplicate", second_factory)


def test_unknown_extension_lists_sorted_available_names() -> None:
    register_extension("zebra", first_factory)
    register_extension("alpha", second_factory)

    with pytest.raises(
        ValueError,
        match=r"unknown extension 'missing'; available extensions: alpha, zebra",
    ):
        resolve_extension_factories(("missing",))


def test_unknown_extension_reports_empty_registry() -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown extension 'missing'; available extensions: <none>",
    ):
        resolve_extension_factories(("missing",))
