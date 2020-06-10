"""Customized Frozen lake enviroment"""
import sys
from contextlib import closing

from gym.envs.toy_text import discrete
from gym import utils
import numpy as np

from six import StringIO

from rltools.environments.common_frozen import *


class ModifiedFrozenLake(discrete.DiscreteEnv):
    """Customized version of gym environment Frozen Lake"""

    def __init__(
            self, desc=None, map_name="4x4", is_slippery=False, n_action=4,
            cyclic_mode=False, hot_edges=False, never_done=False, 
            goal_attractor=False,
            max_reward=0., min_reward=-1., step_penalization=0.):

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min_reward, max_reward)

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        if step_penalization is None:
            step_penalization = 1. / n_state

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        transition_dynamics = {s : {a : [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == LEFT:
                col = max(col - 1, 0)
            elif action == DOWN:
                row = min(row + 1, nrow - 1)
            elif action == RIGHT:
                col = min(col + 1, ncol - 1)
            elif action == UP:
                row = max(row - 1, 0)
            elif action == STAY:
                pass
            else:
                raise ValueError("Invalid action provided")
            return (row, col)

        def compute_transition_dynamics(action_set):

            prob = 1. / len(action_set)
            restart = letter in b'HG' and cyclic_mode

            for action in action_set:
                if not restart:
                    newrow, newcol = inc(row, col, action)
                    newletter = desc[newrow, newcol]
                    newstate = to_s(newrow, newcol)
                    edge_hit = action != STAY and state == newstate
                    got_burned = edge_hit and hot_edges

                    if letter == b'G' and goal_attractor:
                        newletter = letter
                        newstate = state

                    wall_hit = newletter == b'W'
                    if wall_hit:
                        newletter = letter
                        newstate = state
                    fell_in_hole = newletter == b'H'
                    reached_goal = newletter == b'G'
                    ate_candy = newletter == b'C'
                    step_nail = newletter == b'N'

                    numbers = b'0123456789'
                    newpotential = np.int(newletter) if newletter in numbers else 0

                    done = reached_goal or fell_in_hole or got_burned
                    rew = 0.
                    rew -= step_penalization * (1. - done)
                    rew -= step_penalization * newpotential / 10.
                    rew -= step_nail * step_penalization
                    rew += ate_candy * step_penalization / 2.
                    rew += reached_goal * max_reward
                    rew += fell_in_hole * min_reward
                    rew += got_burned * min_reward

                    done = done and not never_done
                    sat_li.append((prob, newstate, rew, done))
                else:
                    done = False
                    rew = - step_penalization
                    for ini_state, start_prob in enumerate(isd):
                        if start_prob > 0.0:
                            sat_li.append((start_prob, ini_state, rew, done))

                # if (not cyclic_mode) or (cyclic_mode and not done):
                #     done = done and not never_done
                #     sat_li.append((prob, newstate, rew, done))
                # else:
                #     done = done and not never_done
                #     for ini_state, start_prob in enumerate(isd):
                #         if start_prob > 0.0:
                #             sat_li.append((start_prob, ini_state, rew, done))

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)

                for action in all_actions:
                    sat_li = transition_dynamics[state][action]
                    letter = desc[row, col]

                    if is_slippery and action is not STAY:
                        action_set = all_actions - set([action])

                    else:
                        action_set = set([action])

                    compute_transition_dynamics(action_set)

        super(ModifiedFrozenLake, self).__init__(n_state, n_action, transition_dynamics, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        else:
            return None
