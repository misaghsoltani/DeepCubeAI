# Environments

DeepCubeAI ships several puzzle / planning environments and a common abstraction in `deepcubeai/environments/environment_abstract.py` that every environment must implement.

## Built-in Environment Keys

Use these exact strings with `--env` (these are always available):

| Key       | Description                           | Notes                                                  |
| --------- | ------------------------------------- | ------------------------------------------------------ |
| cube3     | 3x3x3 Rubik's Cube                    | Supports reversed search test generation (`--reverse`) |
| sokoban   | Sokoban levels (built-in static data) | Uses bundled `sokoban_data` assets                     |
| digitjump | Digit Jump puzzle (puzzlegen)         | Previously documented as `digit_jump`                  |
| iceslider | Ice Slider puzzle (puzzlegen)         | Previously documented as `ice_slider`                  |

## Lazy discovery and registration

DeepCubeAI uses a small, file-backed registry plus a set of builtin entries. The runtime registry is implemented in `deepcubeai.utils.env_utils` and is exposed via the thin helpers in `deepcubeai.environments`.

The runtime registry behavior in brief:

- Builtins are declared in code and always available.
- User environments are stored in a file-backed registry at `deepcubeai/environments/envs.json` (created when you add entries).
- The CLI and APIs discover environments via `deepcubeai.utils.env_utils.list_environments()` which merges builtins + user entries.

### Registration API

- Explicit registration (class object): `deepcubeai.utils.env_utils.register_environment(key, cls)`
- Lazy registration (module path + optional attribute/class name): `deepcubeai.utils.env_utils.register_lazy(key, module_name, attr=None)`

#### Example (lazy register your external module)

```python
from deepcubeai.utils.env_utils import register_lazy

# This only stores a pointer, module isn't imported yet.
register_lazy('myenv', 'my_pkg.myenv_module', 'MyEnv')

# Later, when someone calls deepcubeai.utils.env_utils.get_environment('myenv')
# the module 'my_pkg.myenv_module' will be imported and the class resolved.
```

You can also add/remove entries from the registry using the CLI subcommands `envs-add` and `envs-remove` (these operate on the file-backed registry). Note: builtins cannot be overwritten or removed via the user registry.

### Top-level convenience wrappers

The package exposes a small set of thin wrappers under `deepcubeai.environments` to keep integrations lightweight and avoid importing environment modules until they are actually needed. These functions delegate to `deepcubeai.utils.env_utils`.

Public wrappers available in `deepcubeai.environments`:

- `add_env(key, module, attr=None, *, env_type='user') -> None`
- `get_environment(key) -> Environment`
- `list_environments() -> list[str]`
- `list_environments_info() -> dict[str, dict[str, str | None]]`
- `remove_env(key) -> None`
- `register_environment(key, cls) -> None` (register class object)
- `register_lazy(key, module_name, attr=None) -> None`

Example usage:

```python
from deepcubeai.environments import add_env, register_lazy, list_environments, get_environment

# register a concrete class immediately (writes to the user registry file)
add_env('myenv', 'my_pkg.myenv_module', 'MyEnv')

# or register lazily by module path & attribute name (also writes via add_env/register_lazy)
register_lazy('myenv_lazy', 'my_pkg.myenv_module', 'MyEnv')

# list known envs cheaply (doesn't import individual env modules)
print(list_environments())

# instantiate an environment when needed
env = get_environment('myenv')
```

Why lazy loading? It keeps CLI startup and lightweight tooling fast and prevents module-level side effects from running unless the environment is actually used.

## Abstraction Overview

`Environment` and `State` form the core interface. Important methods / properties to implement (see `deepcubeai.environments.environment_abstract.Environment` / `State` for exact signatures):

| Component                                                         | Required                | Purpose                                               |
| ----------------------------------------------------------------- | ----------------------- | ----------------------------------------------------- |
| `State.__hash__` / `State.__eq__`                                 | Yes                     | Enable hashing & equality for explored sets / caching |
| `State.get_opt_path_len()`                                        | Optional                | If known optimal path length is available             |
| `State.get_solution()`                                            | Optional                | Provide explicit optimal action list                  |
| `Environment.env_name` (property)                                 | Yes                     | Returns the string key (e.g., `cube3`)                |
| `Environment.num_actions_max` (property)                          | Yes                     | Upper bound of branching factor                       |
| `Environment.next_state(states, actions)`                         | Yes                     | Vectorized step function (may mutate inputs)          |
| `Environment.rand_action(states)`                                 | Yes                     | Sample valid random action per state                  |
| `Environment.is_solved(states, states_goal)`                      | Yes (staticmethod)      | Batch solved check vs goal states                     |
| `Environment.state_to_real(states)`                               | Yes                     | Convert states -> observation array (float32)         |
| `Environment.get_env_nnet()` /                                    |                         | Provide (uninitialized) model architectures           |
| `Environment.get_env_nnet_cont()`                                 | Yes                     |                                                       |
| `Environment.get_encoder()` / `get_decoder()`                     | Yes                     | For discrete world model latent mapping               |
| `Environment.get_dqn()`                                           | Yes                     | Network used by heuristic DQN training                |
| `Environment.generate_start_states(num_states, level_seeds=None)` | Yes                     | Produce initial states. May accept `level_seeds`      |
| `Environment.get_goals(states, num_steps)`                        | Optional (staticmethod) | Custom goal derivation if not simple walk             |

See the base class for precise signatures and docstrings.

### Episode Generation Flow

The pipeline uses `Environment.generate_episodes`, which internally:

1. Calls your `generate_start_states` (if your implementation accepts a `level_seeds` argument the pipeline will compute and pass per-trajectory seeds for stratified generation).
2. Performs a random walk using repeated calls to `rand_action` + `next_state` until desired trajectory lengths are reached.
3. Derives `states_start` and `states_goal` as the first / last states.

If your environment supports distinct goal construction logic, implement the optional `get_goals` staticmethod and call it where needed.

### Neural Network Contracts

Return new, uninitialized `torch.nn.Module` instances for each getter. The training scripts will own initialization & optimization. Keep architectures device-agnostic (do not `.to(device)` inside the constructor, the training loop manages that).

### Action Spaces

The framework assumes a discrete action space `[0, num_actions_max)`.

### Adding a New Environment

1. Create `deepcubeai/environments/<your_env>.py`.
2. Define a `State` subclass (immutable or effectively immutable) and an `Environment` subclass.
3. Ensure the `Environment.env_name` property returns the _key_ you'll pass via `--env`.
4. Implement all required abstract methods. Follow tensor / ndarray dtypes used elsewhere (the framework expects `float32` arrays from `state_to_real`).
5. Register the environment so it is discoverable by the CLI/registry:

   - Programmatically: call `deepcubeai.environments.register_lazy(key, module, attr)` or `register_environment(key, cls)` from your integration code.
   - CLI: run `deepcubeai envs-add --key <key> --module <module.path> [--attr <ClassName>]` to write a user entry into `deepcubeai/environments/envs.json`.
   - Note: builtins cannot be overwritten. `envs-add` will raise if you try to replace a builtin key.
