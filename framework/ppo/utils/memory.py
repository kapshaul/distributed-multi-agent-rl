import torch


# Storage / Rollout Buffer
class RolloutBuffer:
    """
    Stores a single batch of transitions
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_advantages_returns(self, last_value, gamma, lam):
        """
        Compute GAE (Generalized Advantage Estimation).
        """
        values = torch.tensor(self.values + [last_value])
        rewards = torch.tensor(self.rewards)
        masks = 1 - torch.tensor(self.dones, dtype=torch.float32)

        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lam * masks[step] * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = values[:-1] + advantages
        return advantages, returns