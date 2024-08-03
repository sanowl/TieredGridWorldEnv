import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, LSTM, Reshape, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import networkx as nx
from scipy.stats import entropy
import plotly.graph_objects as go
from IPython.display import clear_output

class HyperAdvancedMultiAgentEnvironment(gym.Env):
    def __init__(self, grid_size=(20, 20, 5), num_agents=10, num_goals=5, num_obstacles=20):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles
        self.max_steps = 500
        self.current_step = 0
        
        self.action_space = spaces.Discrete(7)  # 6 directions + stay
        self.observation_space = spaces.Box(low=0, high=max(grid_size), 
                                            shape=(10 + num_agents * 4,), dtype=np.float32)
        
        self.weather_conditions = ['clear', 'rainy', 'foggy', 'windy']
        self.current_weather = 'clear'
        self.weather_change_probability = 0.05
        
        self.resource_types = ['food', 'water', 'fuel']
        self.resource_locations = {}
        
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size)
        self.agents = [self._random_position() for _ in range(self.num_agents)]
        self.goals = [self._random_position() for _ in range(self.num_goals)]
        self.obstacles = [self._random_position() for _ in range(self.num_obstacles)]
        self.dynamic_obstacles = [self._random_position() for _ in range(self.num_obstacles // 2)]
        
        for pos in self.obstacles + self.dynamic_obstacles:
            self.grid[pos] = -1
        for pos in self.goals:
            self.grid[pos] = 1
        
        self.current_step = 0
        self.agent_energy = [100] * self.num_agents
        self.agent_resources = [{r: 0 for r in self.resource_types} for _ in range(self.num_agents)]
        
        self._place_resources()
        
        return self._get_obs()

    def _random_position(self):
        while True:
            pos = tuple(random.randint(0, dim-1) for dim in self.grid_size)
            if self.grid[pos] == 0:
                return pos

    def _place_resources(self):
        for r in self.resource_types:
            self.resource_locations[r] = [self._random_position() for _ in range(5)]

    def step(self, actions):
        self.current_step += 1
        rewards = []
        dones = []
        infos = []

        # Update weather
        if random.random() < self.weather_change_probability:
            self.current_weather = random.choice(self.weather_conditions)

        # Move dynamic obstacles
        self._move_dynamic_obstacles()

        for i, action in enumerate(actions):
            old_pos = self.agents[i]
            new_pos = self._get_new_position(old_pos, action)
            
            reward, energy_change, resources_collected = self._get_reward(new_pos, old_pos)
            
            self.agents[i] = new_pos
            self.agent_energy[i] += energy_change
            for r, amount in resources_collected.items():
                self.agent_resources[i][r] += amount
            
            rewards.append(reward)
            dones.append(new_pos in self.goals or self.agent_energy[i] <= 0)
            infos.append({
                'energy': self.agent_energy[i],
                'resources': self.agent_resources[i],
                'weather': self.current_weather
            })

        all_done = all(dones) or self.current_step >= self.max_steps
        return self._get_obs(), rewards, all_done, infos

    def _get_new_position(self, old_pos, action):
        directions = [
            (0, 0, 1), (0, 0, -1),  # up, down
            (0, 1, 0), (0, -1, 0),  # forward, backward
            (1, 0, 0), (-1, 0, 0),  # right, left
        ]
        if action < 6:
            new_pos = tuple(np.clip(old_pos[i] + directions[action][i], 0, self.grid_size[i]-1) for i in range(3))
        else:
            new_pos = old_pos
        return new_pos if self.grid[new_pos] != -1 else old_pos

    def _get_reward(self, pos, old_pos):
        base_reward = -0.1  # Small penalty for each step
        energy_change = -1  # Base energy cost per step
        resources_collected = {r: 0 for r in self.resource_types}

        if pos in self.goals:
            base_reward += 50
            energy_change -= 10
        elif pos in self.obstacles + self.dynamic_obstacles:
            base_reward -= 20
            energy_change -= 20
        elif pos in self.agents:
            base_reward -= 10
            energy_change -= 10
        
        # Collect resources
        for r, locations in self.resource_locations.items():
            if pos in locations:
                base_reward += 5
                resources_collected[r] += 1
                locations.remove(pos)
                locations.append(self._random_position())

        # Weather effects
        if self.current_weather == 'rainy':
            energy_change -= 2
        elif self.current_weather == 'windy':
            if random.random() < 0.2:  # 20% chance to be pushed by wind
                pos = old_pos

        # Distance-based reward
        goal_distances = [np.linalg.norm(np.array(pos) - np.array(goal)) for goal in self.goals]
        min_distance = min(goal_distances)
        base_reward += 1 / (min_distance + 1)  # Avoid division by zero

        return base_reward, energy_change, resources_collected

    def _move_dynamic_obstacles(self):
        for i, obs in enumerate(self.dynamic_obstacles):
            self.grid[obs] = 0
            new_obs = self._get_new_position(obs, random.randint(0, 5))
            self.dynamic_obstacles[i] = new_obs
            self.grid[new_obs] = -1

    def _get_obs(self):
        obs = np.array(self.grid_size + (self.current_step / self.max_steps,) + 
                       tuple(sum(self.agents, ())) + tuple(self.agent_energy))
        for goal in self.goals:
            obs = np.concatenate((obs, goal))
        weather_encoding = [self.current_weather == w for w in self.weather_conditions]
        obs = np.concatenate((obs, weather_encoding))
        return obs

    def render(self, mode='human'):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot grid
        x, y, z = np.meshgrid(range(self.grid_size[0]), range(self.grid_size[1]), range(self.grid_size[2]))
        ax.scatter(x, y, z, c=self.grid.flatten(), cmap='coolwarm', alpha=0.1)
        
        # Plot agents
        agent_positions = np.array(self.agents)
        ax.scatter(agent_positions[:, 0], agent_positions[:, 1], agent_positions[:, 2], c='g', s=100, label='Agents')
        
        # Plot goals
        goal_positions = np.array(self.goals)
        ax.scatter(goal_positions[:, 0], goal_positions[:, 1], goal_positions[:, 2], c='y', s=100, label='Goals')
        
        # Plot obstacles
        obstacle_positions = np.array(self.obstacles + self.dynamic_obstacles)
        ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], obstacle_positions[:, 2], c='r', s=100, label='Obstacles')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f"Step: {self.current_step}/{self.max_steps}, Weather: {self.current_weather}")
        plt.show()

class MetaLearningAgent:
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.meta_learning_rate = 0.001
        self.inner_learning_rate = 0.01
        self.meta_optimizer = Adam(learning_rate=self.meta_learning_rate)
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(256, activation='relu')(input_layer)
        x = Dense(256, activation='relu')(x)
        x = Reshape((1, 256))(x)
        x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = LayerNormalization()(x)
        x = LSTM(128)(x)
        value = Dense(2)(x)  # Two objectives: reward and energy
        advantage = Dense(self.action_size * self.num_agents * 2)(x)
        output = Lambda(lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)))([value, advantage])
        return Model(inputs=input_layer, outputs=output)

    def meta_update(self, task_gradients):
        self.meta_optimizer.apply_gradients(zip(task_gradients, self.model.trainable_variables))

    def adapt(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            action_q_values = tf.gather(q_values, actions, batch_dims=1)
            next_q_values = self.model(next_states)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - action_q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients

    def act(self, state):
        state = np.reshape(state, [1, -1])
        q_values = self.model.predict(state)[0]
        return [np.argmax(q_values[i*self.action_size:(i+1)*self.action_size]) for i in range(self.num_agents)]

class HierarchicalAgent:
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.high_level_policy = self._build_high_level_policy()
        self.low_level_skills = [self._build_low_level_skill() for _ in range(5)]  # 5 different skills

    def _build_high_level_policy(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(256, activation='relu')(input_layer)
        x = Dense(256, activation='relu')(x)
        output = Dense(5, activation='softmax')(x)  # 5 skills to choose from
        return Model(inputs=input_layer, outputs=output)

    def _build_low_level_skill(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.action_size, activation='softmax')(x)
        return Model(inputs=input_layer, outputs=output)

    def act(self, state):
        state = np.reshape(state, [1, -1])
        skill_probs = self.high_level_policy.predict(state)[0]
        chosen_skill = np.random.choice(5, p=skill_probs)
        action_probs = self.low_level_skills[chosen_skill].predict(state)[0]
        return [np.random.choice(self.action_size, p=action_probs) for _ in range(self.num_agents)]

class CommunicatingAgent:
    def __init__(self, state_size, action_size, num_agents, message_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.message_size = message_size
        self.model = self._build_model()

    def _build_model(self):
        state_input = Input(shape=(self.state_size,))
        message_input = Input(shape=(self.num_agents, self.message_size))
        
        x = Dense(256, activation='relu')(state_input)
        x = Dense(256, activation='relu')(x)
        
        message_embedding = Dense(64, activation='relu')(message_input)
        message_embedding = tf.reduce_mean(message_embedding, axis=1)
        
        combined = Concatenate()([x, message_embedding])
        x = Dense(256, activation='relu')(combined)
        
        action_output = Dense(self.action_size, activation='softmax')(x)
        message_output = Dense(self.message_size, activation='tanh')(x)
        
        return Model(inputs=[state_input, message_input], outputs=[action_output, message_output])

    def act(self, state, messages):
        state = np.reshape(state, [1, -1])
        messages = np.reshape(messages, [1, self.num_agents, self.message_size])
        action_probs, new_message = self.model.predict([state, messages])
        return np.argmax(action_probs[0]), new_message[0]

class CuriosityDrivenExploration:
    def __init__(self, state_size):
        self.state_size = state_size
        self.forward_model = self._build_forward_model()
        self.inverse_model = self._build_inverse_model()

    def _build_forward_model(self):
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(1,))
        x = Concatenate()([state_input, action_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        next_state_pred = Dense(self.state_size)(x)
        return Model(inputs=[state_input, action_input], outputs=next_state_pred)

    def _build_inverse_model(self):
        state_input = Input(shape=(self.state_size,))
        next_state_input = Input(shape=(self.state_size,))
        x = Concatenate()([state_input, next_state_input])
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        action_pred = Dense(1)(x)
        return Model(inputs=[state_input, next_state_input], outputs=action_pred)

    def get_curiosity_reward(self, state, action, next_state):
        state = np.reshape(state, [1, -1])
        action = np.reshape(action, [1, 1])
        next_state = np.reshape(next_state, [1, -1])
        next_state_pred = self.forward_model.predict([state, action])
        curiosity_reward = np.mean(np.square(next_state - next_state_pred))
        return curiosity_reward

    def update(self, state, action, next_state):
        state = np.reshape(state, [1, -1])
        action = np.reshape(action, [1, 1])
        next_state = np.reshape(next_state, [1, -1])
        self.forward_model.fit([state, action], next_state, verbose=0)
        self.inverse_model.fit([state, next_state], action, verbose=0)

class AdvancedMultiAgentSystem:
    def __init__(self, env, num_agents, state_size, action_size):
        self.env = env
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.meta_agent = MetaLearningAgent(state_size, action_size, num_agents)
        self.hierarchical_agent = HierarchicalAgent(state_size, action_size, num_agents)
        self.communicating_agents = [CommunicatingAgent(state_size, action_size, num_agents) for _ in range(num_agents)]
        self.curiosity_module = CuriosityDrivenExploration(state_size)
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.action_size) for _ in range(self.num_agents)]
        meta_actions = self.meta_agent.act(state)
        hierarchical_actions = self.hierarchical_agent.act(state)
        communicating_actions = []
        messages = np.zeros((self.num_agents, 10))  # Assuming message size of 10
        for i, agent in enumerate(self.communicating_agents):
            action, message = agent.act(state, messages)
            communicating_actions.append(action)
            messages[i] = message
        # Combine actions using some strategy (e.g., voting)
        combined_actions = [max(set([m, h, c]), key=[m, h, c].count) 
                            for m, h, c in zip(meta_actions, hierarchical_actions, communicating_actions)]
        return combined_actions

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.meta_agent.model.predict(next_state)[0])
            target_f = self.meta_agent.model.predict(state)
            target_f[0][action] = target
            self.meta_agent.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                curiosity_reward = self.curiosity_module.get_curiosity_reward(state, action, next_state)
                total_reward += reward + curiosity_reward
                self.remember(state, action, reward + curiosity_reward, next_state, done)
                state = next_state
                self.replay()
                self.curiosity_module.update(state, action, next_state)
            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")
            if e % 10 == 0:
                self.visualize_training(e)

    def visualize_training(self, episode):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        self.plot_agent_positions()
        plt.subplot(2, 2, 2)
        self.plot_reward_history()
        plt.subplot(2, 2, 3)
        self.plot_q_values()
        plt.subplot(2, 2, 4)
        self.plot_agent_communication()
        plt.tight_layout()
        plt.savefig(f"training_visualization_episode_{episode}.png")
        plt.close()

    def plot_agent_positions(self):
        positions = np.array(self.env.agents)
        plt.scatter(positions[:, 0], positions[:, 1], c='b', label='Agents')
        goal_positions = np.array(self.env.goals)
        plt.scatter(goal_positions[:, 0], goal_positions[:, 1], c='g', label='Goals')
        obstacle_positions = np.array(self.env.obstacles + self.env.dynamic_obstacles)
        plt.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1], c='r', label='Obstacles')
        plt.title("Agent Positions")
        plt.legend()

    def plot_reward_history(self):
        rewards = [transition[2] for transition in self.memory]
        plt.plot(rewards)
        plt.title("Reward History")
        plt.xlabel("Step")
        plt.ylabel("Reward")

    def plot_q_values(self):
        state = self.env.reset()
        q_values = self.meta_agent.model.predict(np.array([state]))[0]
        sns.heatmap(q_values.reshape(self.num_agents, -1), annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Q-values")
        plt.xlabel("Action")
        plt.ylabel("Agent")

    def plot_agent_communication(self):
        messages = np.random.rand(self.num_agents, 10)  # Placeholder for actual messages
        sns.heatmap(messages, annot=True, fmt=".2f", cmap="YlOrRd")
        plt.title("Agent Communication")
        plt.xlabel("Message Dimension")
        plt.ylabel("Agent")

def main():
    env = HyperAdvancedMultiAgentEnvironment()
    num_agents = env.num_agents
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent_system = AdvancedMultiAgentSystem(env, num_agents, state_size, action_size)
    agent_system.train(episodes=1000)

if __name__ == "__main__":
    main()