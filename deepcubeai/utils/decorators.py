from inspect import Parameter, signature
from typing import Any, Callable, Type, TypeVar

T = TypeVar('T', bound=Type[Any])


def optional_abstract_method(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that ensures an abstract method is implemented in a subclass before calling it.

    Args:
        method (Callable[..., Any]): The method to be checked for implementation.

    Returns:
        Callable[..., Any]: The wrapped method that raises an AttributeError if not implemented
            but is called, or the result of the method call if implemented.

    Raises:
        AttributeError: If the method is not implemented in the subclass, but it's called.
    """

    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        """Checks if the method is implemented in the subclass.

        Args:
            instance (Any): The instance of the class.
            *args (Any): Positional arguments passed to the method.
            **kwargs (Any): Keyword arguments passed to the method.

        Returns:
            Any: The result of the method call if implemented.

        Raises:
            AttributeError: If the method is not implemented in the subclass.
        """
        if method.__name__ not in instance.__class__.__dict__:
            raise AttributeError(
                f"{method.__name__} is not implemented in {instance.__class__.__name__}")
        return method(instance, *args, **kwargs)

    return wrapper


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

    @classmethod
    def new_init_subclass(cls: Type[Any], **kwargs: Any) -> None:
        """Override for `__init_subclass__` to enforce default values for `__init__` parameters.

        Args:
            **kwargs (Any): Additional keyword arguments for the subclass initialization.

        Raises:
            TypeError: If any parameter in the `__init__` method does not have a default value.
        """
        original_init_subclass(**kwargs)
        init_signature = signature(cls.__init__)

        for param in init_signature.parameters.values():
            if param.name == 'self':
                continue
            if param.default is Parameter.empty:
                raise TypeError(
                    f"All parameters in the '__init__' method of '{cls.__name__}' must have "
                    f"default values, but parameter '{param.name}' does not.")

    cls.__init_subclass__ = new_init_subclass
    return cls
