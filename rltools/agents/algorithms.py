"""Several Q algorithms"""

import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from rltools.agents.q_table_agent import QTableAgent
from rltools.agents.q_table_agent import EntRegQTableAgent


class SARSA(QTableAgent):
    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, next_action = sarsa_experience

        q_valu = self.q_table[state, action]
        q_next = self.q_table[next_state, next_action]

        q_valu = \
            (1. - self.alpha) * q_valu + \
                  self.alpha  * (reward + self.gamma * q_next * (1.0 - done))

        self._update_q_table(state, action, q_valu)


class EntRegSARSA(EntRegQTableAgent, SARSA):
    pass


class DQN(QTableAgent):
    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, _ = sarsa_experience

        q_valu = self.q_table[state, action]
        q_next = self.q_table[next_state].max()

        q_valu = \
            (1. - self.alpha) * q_valu + \
                  self.alpha  * (reward + self.gamma * q_next * (1.0 - done))

        self._update_q_table(state, action, q_valu)


class EntRegDQN(EntRegQTableAgent, DQN):
    pass


class SoftQ(QTableAgent):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.v_vectr = np.zeros((self.n_states, 1), dtype=np.float)

    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, _ = sarsa_experience

        alpha, gamma = self.alpha, self.gamma
        beta = 1. / self.boltzmann_temperature

        q_valu = self.q_table[state, action]
        v_next = self.v_vectr[next_state]

        q_valu = (1. - alpha) * q_valu + alpha  * (reward + gamma * v_next * (1.0 - done))
        self._update_q_table(state, action, q_valu)

        v_valu = np.log(np.exp(beta * self.q_table[state]).sum()) / beta
        self._update_v_vectr(state, v_valu)

    @property
    def policy(self):
        if self._cached_policy is None:
            policy = np.exp((self.q_table - self.v_vectr) / self.boltzmann_temperature)
            policy /= policy.sum(axis=1).reshape((self.n_states, 1))
            self._cached_policy = policy
        return self._cached_policy

    @property
    def v_vectr(self):
        return self._v_vectr

    @v_vectr.setter
    def v_vectr(self, v_vectr):
        assert v_vectr.shape == (self.n_states, 1)
        self._v_vectr = v_vectr
        self._cached_policy = None

    def _update_v_vectr(self, state, new_value):
        self._v_vectr[state] = new_value
        self._cached_policy = None


class ZLearn(QTableAgent):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.z_vect = np.ones(self.n_states)
        self.rewards = np.zeros(self.n_states)

    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, _ = sarsa_experience
        self.rewards[next_state] = reward

        z_valu = self.z_vect[state]
        z_next = self.z_vect[next_state]

        self.z_vect[state] = \
            (1. - self.alpha) * z_valu + \
            self.alpha  * np.exp(self.rewards[state]) * (z_next ** self.gamma)

        if done:
            self.z_vect[next_state] = \
                (1. - self.alpha) * z_next + \
                    self.alpha * np.exp(self.rewards[next_state])

        q_valu = np.log(self.z_vect[next_state])
        self._update_q_table(state, action, q_valu)


class ZLPlus(QTableAgent):
    def _update_variables(self, sarsa_experience, done):
        state, action, reward, next_state, next_action = sarsa_experience

        q_valu = self.q_table[state, action]
        q_next = self.q_table[next_state, next_action]

        z_valu, z_next, exp_reward = np.exp([q_valu, q_next, reward])

        z_valu = (1. - self.alpha) * z_valu + \
            self.alpha * exp_reward * (z_next ** self.gamma) ** (1. - done)

        q_valu = np.log(z_valu)
        self._update_q_table(state, action, q_valu)


class SoftQDP(SoftQ):
    def __init__(self, *args, dynamics_probs=None, **kvargs):
        super().__init__(*args, **kvargs)

        init_reward = np.max(self.env.reward_range)
        self.rewards = np.empty((self.n_states,), dtype=np.float)
        self.rewards.fill(init_reward)

        self.transition_dynamics = None
        self._init_dynamics(dynamics_probs)

    def _init_dynamics(self, dynamics_probs=None):
        state_action_space = [(s, a)
                              for s in range(self.n_states)
                              for a in range(self.n_actions)]

        nrow = self.n_states
        ncol = self.n_states * self.n_actions

        self.transition_dynamics = lil_matrix((nrow, ncol))
        if dynamics_probs is not None:
            assert isinstance(dynamics_probs, dict)
            for s_i, a_i in state_action_space:
                col = s_i * self.n_actions + a_i
                for prob, s_j, r_j, _ in dynamics_probs[s_i][a_i]:
                    self.transition_dynamics[s_j, col] = prob
                    self.rewards[s_j] = r_j
        else:
            for s_i, a_i in state_action_space:
                col = s_i * self.n_actions + a_i
                prob = 1. / self.n_states
                for s_j in range(self.n_states):
                    self.transition_dynamics[s_j, col] = prob

    def _compute_solution(self, max_step, debug=False):

        beta = 1. / self.boltzmann_temperature
        gamma = self.gamma

        q_table = np.zeros((self.n_states, self.n_actions))
        v_vectr = np.zeros((max_step + 1, self.n_states, 1))

        for steps_left in range(1, max_step + 1):
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    col = state * self.n_actions + action
                    q_update = 0.
                    for next_state in range(self.n_states):
                        row = next_state
                        trans_prob = self.transition_dynamics[row, col]
                        reward = self.rewards[next_state]
                        q_update += \
                            trans_prob * (reward + gamma * v_vectr[steps_left - 1, next_state, 0])
                    q_table[state, action] = q_update
                v_update = np.log((np.exp(beta * q_table[state])).sum()) / beta
                v_vectr[steps_left, state, 0] = v_update

        self.q_table = q_table
        self.v_vectr = v_vectr[max_step]

        if debug:
            plt.plot(range(max_step + 1), v_vectr[:, :, 0])
            plt.show()
