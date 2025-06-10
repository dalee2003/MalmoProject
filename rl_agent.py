# dqn_agent.py

import random
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import constants

class DQNAgent:
    def __init__(self, state_size, action_list,
                 batch_size=constants.BATCH_SIZE,
                 memory_size=constants.MEMORY_SIZE,
                 target_update_freq=constants.TARGET_UPDATE_FREQ,
                 learning_rate=constants.DQN_LEARNING_RATE,
                 gamma=constants.DISCOUNT_FACTOR,
                 epsilon_start=constants.EPSILON_START,
                 epsilon_end=constants.EPSILON_END,
                 epsilon_decay_steps=constants.EPSILON_DECAY_STEPS,
                 debug=constants.DEBUG_AGENT):
        
        self.debug = debug
        
        self.state_size = state_size
        self.action_list = action_list
        self.action_size = len(action_list)
        self.gamma = gamma

        # ε-greedy schedule
        self.epsilon = epsilon_start
        self.eps_min = epsilon_end
        self.eps_decay = (epsilon_start - epsilon_end) / float(epsilon_decay_steps)

        # replay buffer
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # networks
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_network()

        self.train_steps = 0
        self.target_update_freq = target_update_freq

    def _build_model(self, lr):
        m = Sequential()
        m.add(Dense(128, input_dim=self.state_size, activation='relu'))
        m.add(Dense(128, activation='relu'))
        m.add(Dense(self.action_size, activation='linear'))
        m.compile(loss='mse', optimizer=Adam(lr=lr))
        return m

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())
        if self.debug:
            print("Target network synced.")

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis, :])[0]
        return np.argmax(q)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
#         states = np.array([b[0] for b in batch])
#         next_states = np.array([b[3] for b in batch])
#         q_vals = self.model.predict(states)
#         q_next = self.target_model.predict(next_states)

        # Unpack batch and handle None for terminal states
        states = np.array([b[0] for b in batch])
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [np.zeros(self.state_size) if b[3] is None else b[3] for b in batch]
        dones = [b[4] for b in batch]

        next_states = np.array(next_states)

        # Predict Q-values
        q_vals = self.model.predict(states)               # Q(s, ·; θ)
        q_next = self.target_model.predict(next_states)   # Q′(s′, ·; θ⁻)

        for i, (s, a, r, s2, done) in enumerate(batch):
            target = r if done else r + self.gamma * np.max(q_next[i])
            q_vals[i][a] = target

        self.model.fit(states, q_vals, epochs=1, verbose=0)

        # ε decay
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_decay

        # target network update
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()