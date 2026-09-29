import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

from .train_evaluate import PPOTrainer
from .utils.plot import plot_curves_train



def ppo(config):
    # Environment
    ENV_ID = config['ENVIRONMENT']['env_id']
    EVALUATION = config.getboolean('ENVIRONMENT', 'evaluation')
    RENDER = config.getboolean('ENVIRONMENT', 'render')

    # Hyper-parameters
    hyperparameters = {
        "hidden_size": int(config['MODEL']['hidden_size']),
        "learning_rate": float(config['TRAINING']['learning_rate']),
        "gamma": float(config['TRAINING']['gamma']),
        "lambda": float(config['PPO']['lambda']),
        "ppo_clip_eps": float(config['PPO']['ppo_clip_eps']),
        "value_coef": float(config['PPO']['value_coef']),
        "entropy_coef": float(config['PPO']['entropy_coef']),
    }

    # Seed
    seed = int(config['ENVIRONMENT']['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Training configuration
    max_train_steps = int(config['TRAINING']['max_train_steps'])
    max_episode_steps = int(config['TRAINING']['max_episode_steps'])
    batch_size = int(config['TRAINING']['batch_size'])
    ppo_epochs = int(config['PPO']['ppo_epochs'])
    evaluation_episodes = int(config['TRAINING']['evaluation_episodes'])
    num_iterations = int(config['TRAINING']['num_iterations'])

    # CUDA initialization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clears cache to free unused memory
        torch.cuda.ipc_collect()  # Collects unreferenced memory

    # PPO
    ppo = PPOTrainer(ENV_ID, RENDER, hyperparameters)

    # Start to train the PPO agent
    print(f"\n[INFO] Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    train_steps = max_train_steps // num_iterations
    for _ in tqdm(range(num_iterations), desc="Processing"):
        # Train the agent
        ppo.train(train_steps, max_episode_steps, batch_size, ppo_epochs)

    if EVALUATION:
        # Test the agent
        ppo.test(max_episode_steps, evaluation_episodes)

    # Plot the results
    plot_curves_train(ppo.train_reward_log, ppo.train_step_log)
