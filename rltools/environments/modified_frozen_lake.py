"""Customized Frozen lake enviroment"""
import sys
from contextlib import closing

from gym.envs.toy_text import discrete
from gym import utils
import numpy as np

from six import StringIO

class ModifiedFrozenLake(discrete.DiscreteEnv):
    """Customized version of gym environment Frozen Lake"""

    def __init__(
            self, desc=None, map_name="4x4", slippery=0, n_action=4,
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

        if n_action == 2:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = None
        elif n_action == 3:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = 2
        elif n_action in [8, 9]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_leftdown = 4
            a_downright = 5
            a_rightup = 6
            a_upleft = 7
            a_stay = 8
        elif n_action in [4, 5]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 4
        else:
            raise NotImplementedError(f'n_action:{n_action}')

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        if step_penalization is None:
            step_penalization = 1. / n_state

        isd = np.array(desc == b'S').astype('float64').ravel()
        if isd.sum() == 0:
            isd = np.array(desc == b'F').astype('float64').ravel()
        isd /= isd.sum()
        self.isd = isd

        transition_dynamics = {s : {a : [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == a_left:
                col = max(col - 1, 0)
            elif action == a_down:
                row = min(row + 1, nrow - 1)
            elif action == a_right:
                col = min(col + 1, ncol - 1)
            elif action == a_up:
                row = max(row - 1, 0)
            elif action == a_leftdown:
                col = max(col - 1, 0)
                row = min(row + 1, nrow - 1)
            elif action == a_downright:
                row = min(row + 1, nrow - 1)
                col = min(col + 1, ncol - 1)
            elif action == a_rightup:
                col = min(col + 1, ncol - 1)
                row = max(row - 1, 0)
            elif action == a_upleft:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
            elif action == a_stay:
                pass
            else:
                raise ValueError("Invalid action provided")
            return (row, col)

        def compute_transition_dynamics(action_set, action_intended):

            restart = letter in b'HG' and cyclic_mode
            diagonal_mode = n_action in [8, 9]

            for action_executed in action_set:
                prob = 1. / (len(action_set) + slippery)
                prob = (slippery + 1) * prob if action_executed == action_intended else prob

                if not restart:
                    newrow, newcol = inc(row, col, action_executed)
                    newletter = desc[newrow, newcol]
                    newstate = to_s(newrow, newcol)
                    edge_hit = action_executed != a_stay and state == newstate
                    got_burned = edge_hit and hot_edges

                    if letter == b'G' and goal_attractor:
                        newletter = letter
                        newstate = state

                    wall_hit = newletter == b'W'
                    if wall_hit:
                        newletter = letter
                        newstate = state
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'
                    ate_candy = letter == b'C'
                    step_nail = letter == b'N'

                    numbers = b'0123456789'
                    newpotential = np.int(newletter) if newletter in numbers else 0

                    # making diagonal steps costlier makes the agent to avoid them at all,
                    # even if the overall cost of trajectories would be less.
                    # can't yet explain why
                    is_diagonal_step = diagonal_mode and action_executed in [4, 5, 6, 7]
                    diagonal_adjust = 1. if is_diagonal_step else 1.

                    done = is_in_goal or is_in_hole or got_burned
                    rew = 0.
                    rew -= step_penalization * (1. - done) * diagonal_adjust
                    rew -= step_penalization * newpotential / 10.
                    rew -= step_nail * step_penalization / 2.
                    rew += ate_candy * step_penalization / 2.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward
                    rew += got_burned * min_reward

                    done = done and not never_done
                    sat_li.append((prob, newstate, rew, done))
                else:
                    done = False
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'

                    rew = 0.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward

                    for ini_state, start_prob in enumerate(isd):
                        if start_prob > 0.0:
                            sat_li.append((start_prob * prob, ini_state, rew, done))

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)

                for action_intended in all_actions:
                    sat_li = transition_dynamics[state][action_intended]
                    letter = desc[row, col]

                    if slippery != 0:
                        if action_intended == a_left:
                            action_set = set([a_left, a_down, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_down:
                            action_set = set([a_left, a_down, a_right])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_right:
                            action_set = set([a_down, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_up:
                            action_set = set([a_left, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_stay:
                            action_set = set([a_stay])
                        else:
                            raise ValueError(f"encountered undefined action: {action_intended}")

                    else:
                        action_set = set([action_intended])

                    compute_transition_dynamics(action_set, action_intended)

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
