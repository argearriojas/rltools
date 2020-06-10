"""QTableAgent Class"""

import numpy as np
from rltools.agents.agent import Agent

class QTableAgent(Agent):
    """
    Base class for q learning algorithms
    """

    def __init__(self, *args, init_q_vals=None, **kvargs):
        super(QTableAgent, self).__init__(*args, **kvargs)
        self.q_table = np.empty((self.n_states, self.n_actions), dtype=np.float)
        if init_q_vals is None:
            self.q_table.fill(np.max(self.env.reward_range))
        else:
            self.q_table.fill(init_q_vals)
        self._cached_policy = None

    @property
    def policy(self):
        if self._cached_policy is None:
            policy = np.exp(self.q_table / self.boltzmann_temperature)
            policy /= policy.sum(axis=1).reshape((self.n_states, 1))
            self._cached_policy = policy
        return self._cached_policy

    @property
    def q_table(self):
        return self._q_table

    @q_table.setter
    def q_table(self, q_table):
        assert q_table.shape == (self.n_states, self.n_actions)
        self._q_table = q_table
        self._cached_policy = None

    def _update_q_table(self, state, action, new_value):
        self._q_table[state, action] = new_value
        self._cached_policy = None

    @property
    def v_vectr(self):
        return (self.policy * self.q_table).sum(axis=1).reshape((self.n_states, 1))

    @property
    def boltzmann_temperature(self):
        return self._boltzmann_temperature

    @boltzmann_temperature.setter
    def boltzmann_temperature(self, value):
        assert value > 0.
        self._boltzmann_temperature = value
        self._cached_policy = None

    def _action(self, state, greedy):
        q_vals = self.q_table[state]
        if greedy:
            # use greedy policy
            probs = (q_vals == q_vals.max()).astype(np.float)
        else:
            # use boltzmann policy
            probs = np.exp(q_vals / self.boltzmann_temperature)
        probs /= probs.sum()
        return np.random.choice(self.action_set, p=probs)

    def _update_variables(self, sarsa_experience, done):
        raise NotImplementedError


class EntRegQTableAgent(QTableAgent):

    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, next_action = sarsa_experience

        probs = self.policy[state, action]

        #compute before changing q_table/policy
        reward_regularization_term = - np.log(probs) * self.boltzmann_temperature

        reward += reward_regularization_term
        sarsa_experience = state, action, reward, next_state, next_action

        super()._update_variables(sarsa_experience, done)
