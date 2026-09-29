from gymnasium import Wrapper


class FrameSkipWrapper(Wrapper):
    def __init__(self, env, skip=4):
        """A wrapper that skips 'skip' number of frames and accumulates rewards."""
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """Repeats an action for 'self.skip' frames, accumulating rewards."""
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward

            if done:  # If the episode ends, stop skipping
                break

        return obs, total_reward, done, truncated, info