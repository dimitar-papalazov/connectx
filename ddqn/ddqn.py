from keras.models import load_model
import numpy as np
from keras.models import Sequential
from utils import ReplayBuffer, build_dqn


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dimensions, epsilon_decay=0.996,
                 epsilon_end=0.01, memory_size=1000000, fname='ddqn/ddqn_model.h5', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(memory_size, input_dimensions, n_actions, discrete=True)
        self.q_evaluation = build_dqn(alpha, n_actions, input_dimensions, 256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dimensions, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        state = state['board']
        state = np.array(state)
        new_state = new_state['board']
        new_state = np.array(new_state)
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state['board']
        state = np.array(state)
        state = state[np.newaxis, :]
        random_number = np.random.random()
        if random_number < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_evaluation.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_next = self.q_target.predict(new_state)
        q_evaluation = self.q_evaluation.predict(new_state)
        q_predicted = self.q_evaluation.predict(state)
        max_actions = np.argmax(q_evaluation, axis=1)
        q_target = q_predicted.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] * done
        _ = self.q_evaluation.fit(state, q_target, verbose=0)
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
        if self.memory.memory_counter % self.replace_target == 0:
            self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_evaluation.get_weights())

    def save_model(self):
        self.q_evaluation.save(self.model_file)

    def load_model(self):
        self.q_evaluation = load_model(self.model_file)
        if self.epsilon == self.epsilon_min:
            self.update_network_parameters()
