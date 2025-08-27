from __future__ import annotations

from collections.abc import Callable
import contextlib
from inspect import Parameter, signature
from typing import Any, TypeVar

T = TypeVar("T", bound=type[Any])


class OptionalAbstractMethod:
    """A descriptor class that ensures an abstract method is implemented in a subclass before calling it.

    Attributes:
        method (Callable[..., Any]): The abstract method to be checked.
    """

    def __init__(self, method: Callable[..., Any]) -> None:
        """Initializes the OptionalAbstractMethod descriptor.

        Args:
            method (Callable[..., Any]): The abstract method to be checked.
        """
        self.method = method

    def __get__(self, instance: T | None, owner: type[T]) -> Any:
        """Gets the method implementation from the instance or raises an AttributeError if not implemented.

        Args:
            instance (T): The instance of the class.
            owner (Type[T]): The class that owns the method.

        Returns:
            Any: The method implementation if found, otherwise raises an AttributeError.
        """
        if instance is None:
            # Accessed on the class, return the descriptor itself
            return self

        # Check if the method is implemented in the subclass
        if not hasattr(instance.__class__, self.method.__name__):
            raise AttributeError(f"{self.method.__name__} is not implemented in {owner.__name__}")
        return self.method.__get__(instance, owner)


def optional_abstract_method(func: Callable[..., Any]) -> OptionalAbstractMethod:
    """Decorator that ensures an abstract method is implemented in a subclass before calling it.

    Args:
        func (Callable[..., Any]): The abstract method to be checked.

    Returns:
        OptionalAbstractMethod: The descriptor instance that wraps the abstract method.
    """
    return OptionalAbstractMethod(func)


def enforce_init_defaults(cls: T) -> T:
    """Class decorator to enforce default values for all __init__ parameters.

    Ensures that all parameters in the `__init__` method of the subclass inheriting from the
        current class have default values.

    Args:
        cls (Type[T]): The class to be decorated.

    Returns:
        Type[T]: The decorated class with the `__init_subclass__` method overridden.

    Raises:
        TypeError: If any parameter in the `__init__` method does not have a default value.
    """
    original_init_subclass = cls.__init_subclass__

    def new_init_subclass(cls_inner: type[Any], /, **kwargs: Any) -> None:
        """Override for `__init_subclass__` to enforce default values for `__init__` parameters.

        Args:
            cls_inner (type[Any]): The class being initialized.
            **kwargs (Any): Additional keyword arguments for the subclass initialization.

        Raises:
            TypeError: If any parameter in the `__init__` method does not have a default value.
        """
        # Call the original __init_subclass__ if present
        with contextlib.suppress(TypeError):
            original_init_subclass(**kwargs)
        init_signature = signature(cls_inner.__init__)

        for param in init_signature.parameters.values():
            if param.name == "self":
                continue
            if param.default is Parameter.empty:
                raise TypeError(
                    f"All parameters in the '__init__' method of '{cls_inner.__name__}' must have "
                    f"default values, but parameter '{param.name}' does not."
                )

    cls.__init_subclass__ = classmethod(new_init_subclass)  # type: ignore[assignment]
    return cls
