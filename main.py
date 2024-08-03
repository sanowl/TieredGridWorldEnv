import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

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

# Double and Dueling DQN
class DDQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

agent = DDQAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
done = False
batch_size = 32

for e in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, 2])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, 2])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f"episode: {e}/{num_episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

# Plotting the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot([agent.memory[i][2] for i in range(len(agent.memory))])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')

plt.subplot(1, 2, 2)
plt.plot([len(agent.memory[i][0]) for i in range(len(agent.memory))])
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
    state = np.reshape(state, [1, 2])
    action = agent.act(state)
    state, _, done, _ = env.step(action)
    env.render()
