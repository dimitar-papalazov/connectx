from keras.layers import Dense, Activation, Conv2D, BatchNormalization, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.memory_size = max_size
        self.memory_counter = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.memory_size, input_shape))
        self.new_state_memory = np.zeros((self.memory_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, new_states, terminal


def build_dqn(learning_rate, n_actions, input_dimensions, fc1_dimensions, fc2_dimensions):
    model = Sequential([
        Dense(fc1_dimensions, input_shape=(input_dimensions,)),
        Activation('relu'),
        Dense(fc2_dimensions),
        Activation('relu'),
        Dense(n_actions)])
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    return model


def preprocess_state(state):
    columns = 7
    rows = 6
    state = state['board']
    state = np.array(state)
    state = np.reshape(state, (rows, columns, 1))
    return state

