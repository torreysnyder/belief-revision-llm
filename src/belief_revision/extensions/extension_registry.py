"""Named extension factories available to the experiment command line."""

from ..demo import JasonsExperiment
from .extensions import ExtensionFactory
from .torrey.torrey_caa_extension import TorreyCAAExtension
from collections.abc import Iterable, Mapping
from functools import partial

EXTENSION_REGISTRY: dict[str, ExtensionFactory] = {
    "jasons-exp": JasonsExperiment,
    "torrey-caa": TorreyCAAExtension,
}


def register_extension(name: str, factory: ExtensionFactory) -> None:
    """Register one extension factory under a CLI-facing name."""

    if not isinstance(name, str) or not name:
        raise ValueError("extension name must be a non-empty string")
    if name != name.strip():
        raise ValueError("extension name cannot have surrounding whitespace")
    if not callable(factory):
        raise TypeError("extension factory must be callable")
    if name in EXTENSION_REGISTRY:
        raise ValueError(f"extension is already registered: {name!r}")

    EXTENSION_REGISTRY[name] = factory


def resolve_extension_factories(
    names: Iterable[str],
    configurations: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[ExtensionFactory, ...]:
    """Resolve configured CLI factories in order, preserving repetitions."""

    configurations = configurations or {}
    factories: list[ExtensionFactory] = []
    for name in names:
        try:
            factory = EXTENSION_REGISTRY[name]
        except KeyError:
            available = ", ".join(sorted(EXTENSION_REGISTRY)) or "<none>"
            raise ValueError(f"unknown extension {name!r}; available extensions: {available}") from None
        options = configurations.get(name, {})
        factories.append(partial(factory, **options) if options else factory)

    return tuple(factories)
