import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt

class EnhancedTieredGridWorldEnv(gym.Env):
    def __init__(self, grid_size=(5, 5), start=(0, 0), goals=[(4, 4)], obstacles=[(2, 2), (3, 3)]):
        super(EnhancedTieredGridWorldEnv, self).__init__()
        
        self.grid_size = grid_size
        self.start = start
        self.goals = goals
        self.obstacles = obstacles
        self.state = self.start
        
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=max(grid_size)-1, shape=(2,), dtype=np.int32)
        
        # Define reward tiers
        self.r_goal = 10
        self.r_background = -0.1
        self.r_obstacle = -1

    def reset(self):
        self.state = self.start
        return np.array(self.state)

    def step(self, action):
        x, y = self.state
        
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid_size[0] - 1, x + 1)
        
        new_state = (x, y)
        self.state = new_state
        
        reward = self.get_tiered_reward(new_state)
        done = new_state in self.goals
        
        return np.array(self.state), reward, done, {}

    def get_tiered_reward(self, state):
        if state in self.goals:
            return self.r_goal
        elif state in self.obstacles:
            return self.r_obstacle
        else:
            return self.r_background

    def render(self, mode='human'):
        grid = np.zeros(self.grid_size)
        for obstacle in self.obstacles:
            grid[obstacle] = -1
        for goal in self.goals:
            grid[goal] = 10
        grid[self.state] = 1
        print(grid)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Exploration rate decay
num_episodes = 1000  # Number of episodes to train

# Initialize Q-table
env = EnhancedTieredGridWorldEnv()
q_table = np.zeros((*env.grid_size, env.action_space.n))

# Q-learning algorithm with epsilon-greedy strategy
def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state[0], state[1]])  # Exploit

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
    td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
    td_error = td_target - q_table[state[0], state[1], action]
    q_table[state[0], state[1], action] += alpha * td_error

# Training the agent
rewards_per_episode = []
steps_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    total_rewards = 0
    steps = 0
    
    while True:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        update_q_table(state, action, reward, next_state)
        state = next_state
        total_rewards += reward
        steps += 1
        
        if done:
            break
    
    rewards_per_episode.append(total_rewards)
    steps_per_episode.append(steps)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')

plt.subplot(1, 2, 2)
plt.plot(steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

plt.tight_layout()
plt.show()

# Test the trained agent
state = env.reset()
env.render()
done = False

while not done:
    action = np.argmax(q_table[state[0], state[1]])
    state, _, done, _ = env.step(action)
    env.render()
