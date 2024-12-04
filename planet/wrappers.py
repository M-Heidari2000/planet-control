import gym


class RepeatAction(gym.Wrapper):
    """
    Action repeat wrapper to act same action repeatedly
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        return obs, total_reward, done, info
