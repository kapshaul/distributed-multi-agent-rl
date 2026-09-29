# Distributed Deep Q-Learning

This directory implements Deep Q-Networks (DQN) on CartPole. It has two parts:

- an archived notebook containing a **single-process** DQN with recorded outputs;
- a **Ray-based** script that splits experience collection, learning, and evaluation across workers.

The work began as an Oregon State University CS533 course project. The [project page](https://kapshaul.github.io/projects/distributed-agents/) has more background.

Back to the [repository overview](../../README.md). This directory is separate from the PPO code in [`../ppo/`](../ppo/) and does not use the root `main.py` or the root `requirements.txt`.

> **Status:** `distributed_dqn.py` is a prototype entry point. It has not been shown to complete a run in its current form. The [known blockers](#known-blockers-in-distributed_dqnpy) must be addressed before it can train and evaluate end to end. This directory contains no result files, learning curves, or timing measurements from the distributed script.

## DQN in brief

DQN extends Q-learning in three ways:

1. **Function approximation.** A neural network $Q_\theta(s, a)$ replaces the Q-table.
2. **Experience replay.** Transitions $(s, a, r, s', d)$ are stored in a buffer, and updates use random minibatches sampled from it. This reuses experience and weakens the correlation between consecutive updates.
3. **A target network.** A second network with parameters $\theta^-$ supplies the bootstrap value and is periodically overwritten with the online parameters, $\theta^- \leftarrow \theta$.

For a sampled transition, the code regresses $Q_\theta(s, a)$ toward

$$
y =
\begin{cases}
r, & \text{if } s' \text{ is terminal},\\
r + \beta \max_{a'} Q_{\theta^-}(s', a'), & \text{otherwise,}
\end{cases}
$$

where $\beta$ is the discount factor (`beta = 0.99` in the code). The loss is `SmoothL1Loss` (Huber), optimized with Adam. When `use_target_model` is `False`, the target computation is written to use the online network $Q_\theta$ in place of $Q_{\theta^-}$. The full no-target configuration still fails, however, because `learn` accesses `self.target_model` unconditionally (see [known blockers](#known-blockers-in-distributed_dqnpy)), so it is not a working ablation.

Episodes that reach the 200-step cap without the pole falling are *not* flagged as terminal. For those transitions the target still bootstraps from $s'$.

## Architecture of `distributed_dqn.py`

<img src="architecture.png" alt="Distributed DQN architecture with collectors, evaluator, model server, and memory server" width="800">

| Component | Ray construct | What the code does |
| --- | --- | --- |
| **Model server** | `@ray.remote` actor `model_server` | Holds the online network (`eval_model`) and the target network. Chooses every action on behalf of the collectors (`explore_or_exploit_policy`, with ε decaying linearly from 1 to `final_epsilon`). When a collector reports a finished episode of *n* steps, `learn(n)` advances the server's step counter *n* times. It samples a minibatch and updates the online network every `update_steps` steps, and copies the online weights into the target network every `model_replace_freq` steps. It also hands out evaluation work and saves the best model. |
| **Memory server** | `@ray.remote` actor `ReplayBuffer_remote` | A FIFO ring buffer of 2000 transitions (hard-coded in `distributed_DQN_agent`). Minibatches are sampled uniformly with replacement. |
| **Collectors** (default 4) | `@ray.remote` task `collecting_worker` | Each collector runs its own CartPole copy. For every step it asks the model server for an action, then pushes the transition to the memory server. At the end of each episode it calls `learn` and stops once `learn` returns `True`. |
| **Evaluators** (default 4) | `@ray.remote` task `evaluation_worker` | Each evaluator polls `ask_evaluation`. When given a checkpoint index, it runs 30 greedy episodes and writes the average return. |

Two consequences of this design:

- **Learning happens in bursts at episode boundaries.** All gradient updates run on the model-server actor.
- **Every environment step waits on a remote call to the server.** The distributed version helps only when stepping the environment is slow relative to that round trip. For this reason the script uses [`custom_cartpole.py`](custom_cartpole.py), a standard CartPole with `time.sleep(0.01)` added to every step.

## Files

| File | Role |
| --- | --- |
| [`distributed_dqn.py`](distributed_dqn.py) | Ray script: model server, collectors, evaluators, `distributed_DQN_agent`, and `main()` |
| [`dqn_model.py`](dqn_model.py) | `_DQNModel`: MLP with layers 4 → 256 → 64 → 2, tanh activations, and Xavier initialization. `DQNModel` wraps it in `nn.DataParallel` and provides `predict`, `predict_batch`, `fit` (Huber loss + Adam), `replace`, `save`, and `load`. |
| [`memory_remote.py`](memory_remote.py) | Ray-actor replay buffer used by the script |
| [`memory.py`](memory.py) | Local replay buffer used by the notebook's single-process agent |
| [`custom_cartpole.py`](custom_cartpole.py) | Classic CartPole with a 10 ms sleep per step, a 200-step cap (`_max_episode_steps`), and the old Gym `step()` signature returning 4 values |
| [`distributed_dqn.ipynb`](distributed_dqn.ipynb) | Archived course notebook (see below) |
| [`requirements.txt`](requirements.txt) | Unpinned dependency list for this directory |
| [`architecture.png`](architecture.png) | Architecture diagram |

## Default settings in the script

| Setting | Value | Where it is set |
| --- | --- | --- |
| Training episodes / evaluation interval | 10,000 / every 50 episodes | `main()` |
| Evaluation trials per checkpoint | 30 | `main()` |
| Collectors / evaluators | 4 / 4 | `distributed_DQN_agent.__init__` defaults |
| ε schedule | 1 → 0.1, linear over 100,000 server steps | `hyperparams_CartPole` |
| Minibatch size / update period | 32 / every 10 steps | `hyperparams_CartPole` |
| Target replacement period | every 2000 steps | `hyperparams_CartPole` |
| Replay capacity | 2000 | Hard-coded in `ReplayBuffer_remote.remote(2000)`. The `memory_size` hyperparameter is not read. |
| Discount β / learning rate | 0.99 / 3e-4 | `hyperparams_CartPole` |
| Ray init | `include_webui=False, redis_max_memory=5e8, object_store_memory=5e9` | Module level |

The model server always receives the module-level `hyperparams_CartPole` dictionary, whatever is passed to `distributed_DQN_agent`. The `update_steps` and `model_replace_freq` locals inside `collecting_worker` are never used.

## The archived notebook and the script

[`distributed_dqn.ipynb`](distributed_dqn.ipynb) is the course notebook.

- **Part 1** runs a *single-process* `DQN_agent` on `gym.make('CartPole-v0')` with the local `ReplayBuffer`. It uses the same hyperparameters and trains for 10,000 episodes, evaluating every 50. The notebook metadata records Python 3.7.11, and saved `pip` output shows `gym` 0.21.0 and `torch` 1.10.0. The notebook stores 405 cell outputs, 402 of them in cell 20. Part 1 also lists the replay/target ablations that the course assignment requested.
- **Part 2** only initializes Ray and describes the distributed design. It notes that the distributed agent was meant to run as a standalone script on a compute node, and the notebook contains no distributed implementation.

The distributed implementation exists only in `distributed_dqn.py`. Results in the notebook come from the single-process agent and do not measure the Ray version.

## Running the script (conditional)

The intended invocation is shown below. The bare imports of sibling modules (`from dqn_model import ...`) resolve from any working directory, because Python puts the script's directory on `sys.path`. The output paths, however, are relative to the working directory, so run it from this directory to keep them here:

```bash
cd framework/distributed_dqn
pip install -r requirements.txt   # unpinned; see the dependency notes below
python distributed_dqn.py
```

These commands will not complete a training run until the blockers below are resolved. When the script does run, it writes the following paths relative to the working directory (this directory in the commands above):

- `CartPole-v0/` is created at import time.
- `CartPole-v0/result_file_1.txt` receives one average evaluation return per line, followed by the total wall-clock time.
- `CartPole-v0/best_model.pt` holds the best-scoring online network.
- `reward_4cv_4ev.png` is the learning curve, plotted against the episode index.

## Known blockers in `distributed_dqn.py`

- **Undefined `training_episodes` in `model_server.learn`.** The method compares `self.episode` against a module-level name `training_episodes`. That name exists only as a local inside `main()`, so the first `learn` call raises `NameError`. The failure reaches each collector through `ray.get`, and the collectors stop. The evaluators wait until `self.episode >= self.training_episodes`, which then never happens, so they can poll indefinitely and `ray.wait` never returns. Reading `self.training_episodes` would match the rest of the class.
- **Historical Ray arguments.** `ray.init(include_webui=..., redis_max_memory=...)` uses keyword arguments from early Ray releases that later versions reject. `requirements.txt` does not pin Ray.
- **`use_target_model=False` still accesses the target network.** `learn` calls `self.target_model.replace(...)` unconditionally, but `target_model` is created only when `use_target_model` is `True`. The no-target ablation therefore fails with `AttributeError`.
- **Evaluator skips checkpoint 0 and evaluates the live network.** `evaluation_worker` tests `if not num`, which treats index 0 like "no work", so `results[0]` is never filled and stays 0. `privous_q_net` stores references to the same `eval_model` object rather than copies. Evaluators also call `server.greedy_policy`, which uses the *current* network. Each reported value is therefore the performance of the network at evaluation time, not a frozen snapshot from the corresponding 50-episode boundary.
- **Missing and extra dependencies.** The script imports `IPython.display.HTML`, but `IPython` is not in `requirements.txt`. The script does not use `JSAnimation` or the `Box2D` extra of `gym`, although the requirements file lists them.
- **NumPy 2 incompatibility in the replay buffer.** `memory_remote.py` builds arrays with `np.array(action, copy=False)` from Python integers. Under NumPy 2 this raises `ValueError` because a copy is required, and the unpinned `numpy` requirement does not rule NumPy 2 out.
- **Old Gym API.** `custom_cartpole.py` subclasses `gym.Env` and returns 4-tuple steps. The script expects that interface rather than the Gymnasium 5-tuple interface.

## Dependency notes

This directory's [`requirements.txt`](requirements.txt) lists `gym[Box2D]`, `torch`, `JSAnimation`, `matplotlib`, `ray`, `tqdm`, and `numpy`, with no versions pinned. It is separate from the repository-root `requirements.txt`, which is a UTF-16 CUDA 12.8 snapshot for PPO that does not include Ray or Gym.

A working environment needs:

- `IPython`;
- a Ray version that accepts the `ray.init` arguments above, or an edited call;
- a NumPy version compatible with `copy=False`, or an edited buffer.

The notebook metadata and saved outputs point to Python 3.7.11, gym 0.21.0, and torch 1.10.0 as the original environment, but they do not record the original Ray version.
