import importlib.util
import inspect
import os
import sys
from typing import Type, Union

from deepcubeai.environments.environment_abstract import Environment


def find_env_class_by_name(env_name: str) -> Union[Type[Environment], None]:
    """Finds and returns the environment class by its name.

    Args:
        env_name (str): The name of the environment to find.

    Returns:
        Union[Type[Environment], None]: The environment class if found, otherwise None.
    """
    env_name_lower = env_name.lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.abspath(os.path.join(script_dir, '..'))
    environments_dir = os.path.join(root_directory, 'environments')

    for filename in os.listdir(environments_dir):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            file_path = os.path.join(environments_dir, filename)

            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if hasattr(obj, 'get_env_name'):
                        method = getattr(obj, 'get_env_name')
                        if callable(method) and not is_abstract_method(obj, 'get_env_name'):
                            env_class_name = obj().get_env_name().lower()
                            if env_class_name == env_name_lower:
                                sys.modules[obj.__module__] = module
                                return obj  # Return the class if the environment name matches
            except Exception:
                continue  # Skip to the next file in case of an error

    return None  # Return None if no matching environment is found


def is_abstract_method(cls: Type, method_name: str) -> bool:
    """Checks if a method is abstract in a given class.

    Args:
        cls (Type): The class to check.
        method_name (str): The name of the method to check.

    Returns:
        bool: True if the method is abstract, False otherwise.
    """
    return bool(
        getattr(cls, '__abstractmethods__', None) and method_name in cls.__abstractmethods__)


def get_environment(env_name: str) -> Environment:
    """Gets an instance of the environment by its name.

    Args:
        env_name (str): The name of the environment to get.

    Returns:
        Environment: An instance of the environment.

    Raises:
        ValueError: If no known environment is found with the given name.
    """
    env_name = env_name.lower()
    env_class = find_env_class_by_name(env_name)

    if env_class:
        env = env_class()
        return env
    else:
        raise ValueError(f"No known environment {env_name}")
