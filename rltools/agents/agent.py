"""Agent Class"""

import numpy as np
from rltools.utils.visualization import plot_dist

class Agent:
    """The Base class for RL Agents"""

    def __init__(self, env, alpha=0.5, gamma=1., boltzmann_temperature=0.2,
                 batch_size=1):
        self.env = env
        self.boltzmann_temperature = boltzmann_temperature
        self.alpha, self.gamma = alpha, gamma
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.action_set = np.array(list(range(self.n_actions)))
        self.batch_size = batch_size
        self._replay_memory = []

    @property
    def boltzmann_temperature(self):
        return self._boltzmann_temperature

    @boltzmann_temperature.setter
    def boltzmann_temperature(self, value):
        assert value > 0.
        self._boltzmann_temperature = value

    def train_and_evaluate(self, n_episodes=100):
        """This method is used with joblib for training and evaluating several
        agents in parallel"""
        self.train(n_episodes)
        avg_reward_greed = self.evaluate()
        avg_reward_stoch = self.evaluate(greedy=False)
        return self, avg_reward_greed, avg_reward_stoch

    def train(self, n_episodes=100):
        """Train the agent"""
        return self._train(n_episodes)

    def evaluate(self, n_episodes=30, greedy=True, show_dist=False,
                 use_log=True):
        """evaluate the agent"""
        return self._evaluate(n_episodes, greedy, show_dist, use_log)

    def _save_experience(self, sarsa_experience, done):
        self._replay_memory.append((sarsa_experience, done))

    def _run_episode(self, track_experience, greedy):
        env = self.env
        # keep SARSA architecture
        state, done = env.reset(), False
        action = self._action(state, greedy=greedy)
        episode_reward = 0
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = self._action(next_state, greedy=greedy)
            if track_experience:
                sarsa_experience = \
                    (state, action, reward, next_state, next_action)
                self._save_experience(sarsa_experience, done)
            state, action = next_state, next_action
            episode_reward += reward
        return episode_reward

    def _learn(self):
        # np.random.shuffle(self._replay_memory)
        for sarsa_experience, done in self._replay_memory:
            self._update_variables(sarsa_experience, done)
        self._replay_memory.clear()

    def _train(self, n_episodes):
        for i in range(1, n_episodes + 1):
            self._run_episode(track_experience=True, greedy=False)
            if i % self.batch_size == 0:
                self._learn()

    def _evaluate(self, n_episodes, greedy, show_dist, use_log):
        avg_reward = 0
        for _ in range(n_episodes):
            avg_reward += \
                self._run_episode(track_experience=show_dist, greedy=greedy)
        if show_dist:
            visited = np.zeros(self.env.desc.shape, dtype=np.int).flatten()
            for (_, _, _, obs, _), _ in self._replay_memory:
                visited[obs] += 1
            if use_log:
                visited = np.log(visited + 1)
            plot_dist(self.env.desc, visited)
            self._replay_memory.clear()

        avg_reward /= n_episodes
        return avg_reward

    def _update_variables(self, sarsa_experience, done):
        raise NotImplementedError

    def _action(self, state, greedy):
        raise NotImplementedError
