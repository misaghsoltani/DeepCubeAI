from typing import Any, Callable


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
