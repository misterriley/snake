from collections import deque
from operator import mul
import functools

from keras.layers import Dense, Flatten, Reshape, Concatenate
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras import losses
from tensorflow.keras.models import Model

import numpy as np


class DQNAgent:

    def __init__(self, environment, trained_model = None):
        # Initialize constant
        self.environment = environment
        self.obs_size = environment.observation_space.shape
        self.obs_dim = functools.reduce(mul, self.obs_size)
        self.action_size = environment.action_space.n
        self.consecutive_episodes = 100

        # Hyperparameters of the training
        self.learning_rate = 0.0005
        self.gamma = 0.99  # discount factor
        self.replay_memory = 50000
        self.replay_size = 16

        # Initialize neural network model
        if trained_model:
            self.model = self.load_model(filename = trained_model)
        else:
            self.model = DAQN(environment.observation_space.shape, environment.action_space.n)
            self.model.compile(optimizer = 'adam', loss = losses.MeanSquaredError())

        # Exploration/exploitations parameters
        self.epsilon = .4
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.001
        self.action_b4_replay = 256

        # Define variable
        self.storage = deque(maxlen = self.replay_memory)
        self.sum_reward, self.rewards_lst = 0.0, []

    def store(self, state, action, reward, next_state, done):
        # Save history to storage for replay

        self.storage.append(StorageData(state, action, reward, next_state, done))

    def action(self, state, reward, done, episode, action_count, training = True):
        # Update cumulative reward
        self.sum_reward += reward
        cnt_state = np.asarray([state.reshape((*state.shape, 1))])  # for the Conv2D nets where there is 1 "channel"

        # Episode ends
        if done:
            self.rewards_lst.append(self.sum_reward)
            avg_reward = np.mean(self.rewards_lst[-self.consecutive_episodes: ])
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)
            print (f'Episode {episode}, Reward: {self.sum_reward:.2f}, Average rewards {avg_reward:.5f}, Epsilon {self.epsilon:.2f}')
            self.sum_reward = 0.0

            return -1

        # Episode not ends: return next action
        else:
            # Train agent
            if training:
                if action_count >= self.action_b4_replay:
                    if action_count % self.replay_size == 0:
                        self.replay()
                    if np.random.random() < self.epsilon:
                        action = self.environment.action_space.sample()
                    else:
                        act_values = self.model.predict(cnt_state)[:action_count]
                        action = np.argmax(act_values[0])
                        # print("best Q: " + str(act_values[0][action]))
                else:
                    action = self.environment.action_space.sample()

            # Run trained agent
            else:
                act_values = self.model.predict(cnt_state)
                action = np.argmax(act_values[0])

            return action

    def replay(self):
        minibatch_idx = np.random.permutation(len(self.storage))[: self.replay_size]

        states = np.asarray([self.storage[i].state.flatten() for i in minibatch_idx])
        actions = np.asarray([self.storage[i].action for i in minibatch_idx])
        rewards = np.asarray([self.storage[i].reward for i in minibatch_idx])
        next_states = np.asarray([self.storage[i].next_state.flatten() for i in minibatch_idx])
        dones = np.asarray([self.storage[i].done for i in minibatch_idx])

        X_batch = states
        Y_batch = np.zeros(shape = (self.replay_size, self.action_size + self.obs_dim), dtype = np.float64)

        qValues_batch = self.model.predict(states)
        qValuesNewState_batch = self.model.predict(next_states)

        targetValue_batch = np.copy(rewards)
        targetValue_batch += (1 - dones) * self.gamma * np.amax(qValuesNewState_batch, axis = 1)

        for idx in range(self.replay_size):
            targetValue = targetValue_batch[idx]
            Y_sample = qValues_batch[idx]
            Y_sample[actions[idx]] = targetValue
            Y_batch[idx] = Y_sample

            if dones[idx]:
                X_batch = np.append(X_batch, np.asarray([next_states[idx]]), axis = 0)
                Y_batch = np.append(Y_batch, np.asarray([[rewards[idx]] * self.action_size]), axis = 0)

        Y_batch = np.append(Y_batch, X_batch)
        self.model.fit(X_batch, Y_batch, batch_size = len(X_batch), epochs = 1, verbose = 2)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        return load_model(filename)


class DAQN(Model):

    def __init__(self, state_shape, action_shape, latent_dim = 12):
        super(DAQN, self).__init__()
        self.state_shape = state_shape
        self.num_inputs = functools.reduce(mul, self.state_shape)
        self.action_shape = action_shape
        self.encoder = Sequential([
          Dense(latent_dim, input_shape = (self.num_inputs,), activation = 'relu'),
        ])
        self.decoder = Sequential([
          Dense(self.num_inputs, activation = 'sigmoid')
        ])
        self.q_network = Sequential([
            Dense(16, activation = 'relu'),
            Dense(self.action_shape, activation = 'linear')
        ])
        self.final_layer = Concatenate()

    def call(self, x):
        encoded = self.encoder(x)
        print(encoded)
        decoded = self.decoder(encoded)
        qs = self.q_network(x)
        return self.final_layer([qs, decoded])


class StorageData:

    def __init__(self, state, action, reward, next_state, done):
        self.state = state.reshape((*state.shape, 1))
        self.action = action
        self.reward = float(reward)
        self.next_state = next_state.reshape((*next_state.shape, 1))
        self.done = done
