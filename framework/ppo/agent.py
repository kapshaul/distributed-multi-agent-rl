import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .model.model import PPONetwork, PPONetwork_CNN
from common import is_image_observation_env



class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size, lr, gamma, lam,
                 ppo_clip_eps, value_coef, entropy_coef, is_image_env, device):
        self.device = device

        # Choose network architecture based on the provided state_dim.
        # Here, if state_dim has length 4, we assume an image-like state input and select a CNN.
        if not is_image_env:
            self.model = PPONetwork(state_dim, action_dim, hidden_size).to(device)
        else:
            self.model = PPONetwork_CNN(state_dim, action_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #layer_lr(optimizer=self.optimizer, model=self.model, target_layer_name="common", lr=lr/10)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.lam = lam
        self.ppo_clip_eps = ppo_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state)
        return action, log_prob, value

    def compute_advantages_returns(self, rollout_buffer, last_value):
        advantages, returns = rollout_buffer.compute_advantages_returns(
            last_value, self.gamma, self.lam
        )
        return advantages, returns

    def update(self, rollout_buffer, advantages, returns, ppo_epochs, minibatch_size):
        b_states = torch.FloatTensor(np.array(rollout_buffer.states)).to(self.device)
        b_actions = torch.LongTensor(np.array(rollout_buffer.actions)).to(self.device)
        b_log_probs = torch.FloatTensor(np.array(rollout_buffer.log_probs)).to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).to(self.device)

        batch_size = b_states.shape[0]
        indices = np.arange(batch_size)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                # Compute Loss
                new_log_probs, entropy, values = self.model.evaluate_actions(mb_states, mb_actions)
                ratio = (new_log_probs - mb_old_log_probs).exp()

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
