import numpy as np
from environment import Env, GraphicDisplay

UNIT = 100
HEIGHT = 5
WIDTH = 5
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]
ACTION = [(-1, 0), (1, 0), (0, -1), (0, 1)]
REWARD = []

class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.value_table = [[0.0] * WIDTH for _ in range(HEIGHT)]
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * WIDTH for _ in range(HEIGHT)]
        self.policy_table[2][2] = []
        self.discount_factor = 0.9

        self.reward = [[0] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[2][2] = 1
        self.reward[1][2] = -1
        self.reward[2][1] = -1
        self.all_state = []
        for x in range(WIDTH):
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def policy_evaluation(self):
        next_value_table = [[0.00] * WIDTH for _ in range(HEIGHT)]

        for state in self.all_state:
            value = 0.0

            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            for action in self.possible_actions:
                next_state = self.state_after_action(state, action)
                reward = self.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = value
            print(next_value_table)
        self.value_table = next_value_table

    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.all_state:
            if state == [2, 2]:
                continue

            value_list = []

            result = [0.0, 0.0, 0.0, 0.0]

            for index, action in enumerate(self.possible_actions):
                next_state = self.state_after_action(state, action)
                reward = self.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            max_idx_list = np.argwhere(value_list == np.amax(value_list))
            max_idx_list = max_idx_list.flatten().tolist()
            prob = 1 / len(max_idx_list)

            for idx in max_idx_list:
                result[idx] = prob

            next_policy[state[0]][state[1]] = result
            print(result)
        self.policy_table = next_policy

    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        return np.random.choice(4, 1, p=policy)[0]

    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTION[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1 if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1 if state[1] > HEIGHT - 1 else state[1])
        return state

if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()