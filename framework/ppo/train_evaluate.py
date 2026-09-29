import os
import torch
import numpy as np
from datetime import datetime

import ale_py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation

from .agent import PPOAgent
from .utils.memory import RolloutBuffer
from common import is_image_observation_env
            


class PPOTrainer:
    def __init__(self, env_id, render, hyperparameters):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            current_device = torch.cuda.current_device()
            gpu_model = torch.cuda.get_device_name(current_device)
            print(f"CUDA is available: True | Using GPU: {gpu_model}")
        else:
            print("CUDA is available: False | Using CPU")
        self.device = torch.device("cuda" if is_cuda else "cpu")

        # Hyper-parameters
        self.env_id = env_id
        gym.register_envs(ale_py)
        self.hidden_size = hyperparameters["hidden_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.gamma = hyperparameters["gamma"]
        self.lam = hyperparameters["lambda"]
        self.ppo_clip_eps = hyperparameters["ppo_clip_eps"]
        self.value_coef = hyperparameters["value_coef"]
        self.entropy_coef = hyperparameters["entropy_coef"]

        # Create environment
        is_image_env = is_image_observation_env(env_id=env_id)
        if not is_image_env:
            self.env = gym.make(
                env_id,
                render_mode="rgb_array" if render else None,
            )
        else:
            self.env = gym.make(
                id=env_id,
                obs_type="grayscale",
                #frame_skip=4,
                render_mode="rgb_array" if render else None,
            )
            self.env = ResizeObservation(self.env, (84, 84))
            self.env = FrameStackObservation(self.env, stack_size=4)

        # Log environment information
        print("===== Environment Information =====")
        print(f"Environment ID: {env_id}")
        print(f"Observation Space: {self.env.observation_space.shape}")
        print(f"Action Space Type: {'Discrete' if isinstance(self.env.action_space, gym.spaces.Discrete) else 'Continuous'}")
        print(
            f"Action Space Size: {self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else self.env.action_space.shape}")
        print(f"Max Episode Steps: {self.env.spec.max_episode_steps if self.env.spec else 'Unknown'}")
        print(f"Environment Metadata: {self.env.metadata}")

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        # Initialize PPO agent
        self.agent = PPOAgent(
            self.state_dim, self.action_dim, self.hidden_size,
            self.learning_rate, self.gamma, self.lam,
            self.ppo_clip_eps, self.value_coef, self.entropy_coef,
            is_image_env, self.device
        )

        self.rollout_buffer = RolloutBuffer()

        self.episode, self.total_steps = 0, 0
        self.train_reward_log, self.test_reward_log = [], []
        self.train_step_log = []

        # Create log directory if it doesn't exist
        log_dir = os.path.join("results", "log")
        os.makedirs(log_dir, exist_ok=True)
        # Format current time (e.g., 2025-06-14_10-30-00)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Set the full path with timestamp
        self.reward_log_path = os.path.join(log_dir, f"reward_{timestamp}.log")

    def train(self, max_train_steps, max_episode_steps, batch_size, ppo_epochs):
        """
        Train
        """
        with open(self.reward_log_path, "w") as f:
            pass

        state, info = self.env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        total_reward, steps, epiosde_steps = 0, 0, 0
        while steps < max_train_steps:
            # Collect data for one batch
            self.rollout_buffer.clear()
            for _ in range(batch_size):
                with torch.no_grad():
                    self.agent.model.eval()
                    action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                total_reward += reward
                self.rollout_buffer.add(
                    state.to(torch.device('cpu')).numpy(),
                    action,
                    log_prob.item(),
                    reward,
                    done,
                    value.item()
                )

                state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
                steps += 1
                epiosde_steps += 1
                self.total_steps += 1

                if done or epiosde_steps >= max_episode_steps:
                    state, info = self.env.reset()
                    state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)

                    print(f"Episode: {self.episode + 1}      Rewards: {total_reward}      Steps: {self.total_steps}")
                    with open(self.reward_log_path, "a") as f:
                        f.write(f"Episode: {self.episode + 1}      Rewards: {total_reward}      Steps: {self.total_steps}\n")
                    self.train_reward_log.append(total_reward)
                    self.train_step_log.append(self.total_steps)
                    self.episode += 1
                    total_reward, epiosde_steps = 0, 0

            # Compute advantages and returns
            with torch.no_grad():
                _, next_value = self.agent.model.forward(state)
                last_value = next_value.item()

                advantages, returns = self.agent.compute_advantages_returns(
                    self.rollout_buffer, last_value
                )

            # PPO update
            self.agent.model.train()
            self.agent.update(self.rollout_buffer, advantages, returns, ppo_epochs, 1024)
        torch.save(self.agent.model.state_dict(), 'model/model.pt')

    def test(self, max_episode_steps, episodes=10):
        """
        Test
        """

        #self.env = gym.make(self.env_id, render_mode="human")

        total_rewards = []
        for episode in range(episodes):
            state, info = self.env.reset()
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < max_episode_steps:
                action, _, _ = self.agent.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)

            total_rewards.append(episode_reward)
            print(f"Evaluation {episode + 1},      Reward = {episode_reward}")

        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward: {avg_reward}\n")
        self.test_reward_log.append(avg_reward)
