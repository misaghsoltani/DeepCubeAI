from environments.environment_abstract import Environment


def get_environment(env_name: str) -> Environment:
    """Get the environment instance based on the environment name.

    Args:
        env_name (str): The name of the environment.

    Returns:
        Environment: An instance of the requested environment.

    Raises:
        ValueError: If the environment name is not recognized.
    """
    env_name = env_name.lower()
    env: Environment

    if env_name == "cube3":
        from environments.cube3 import Cube3
        env = Cube3()

    elif env_name == "sokoban":
        from environments.sokoban import Sokoban
        env = Sokoban(10, 4)

    elif env_name == "cube3_triples":
        from environments.cube3 import Cube3
        env = Cube3(do_action_triples=True)

    elif env_name == "iceslider":
        from environments.ice_slider import IceSliderEnvironment
        env = IceSliderEnvironment()

    elif env_name == "digitjump":
        from environments.digit_jump import DigitJumpEnvironment
        env = DigitJumpEnvironment()

    else:
        raise ValueError(f"No known environment {env_name}")

    return env
