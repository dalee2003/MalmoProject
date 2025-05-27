# rl_agent.py

# --- Make these the VERY FIRST executable lines ---
import os
# print("RL_AGENT: Top of file. Setting CUDA_VISIBLE_DEVICES...") # Optional debug print
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU visibility for TensorFlow
# print("RL_AGENT: CUDA_VISIBLE_DEVICES set. Attempting TensorFlow import...") # Optional debug print
try:
    import tensorflow as tf
    # print("RL_AGENT: TensorFlow imported successfully in rl_agent.py! Version:", tf.__version__) # Optional debug print
except Exception as e:
    print("RL_AGENT: ERROR importing TensorFlow in rl_agent.py!")
    import traceback
    traceback.print_exc()
    raise # Re-raise the exception to stop execution if TF fails to import here
# --- End of critical first import ---

import random
import numpy as np
import json
import collections # For the replay buffer (deque)

try:
    import constants
except ImportError:
    # Fallback dummy constants - Ensure your real constants.py is comprehensive
    class constants:
        DEBUG_AGENT = True
        STATE_SIZE = 17
        LEARNING_RATE = 0.001 # For DQN
        DQN_LEARNING_RATE = 0.001 # Explicit for DQN
        REPLAY_BUFFER_SIZE = 10000
        BATCH_SIZE = 32
        TARGET_NETWORK_UPDATE_FREQUENCY = 100
        DISCOUNT_FACTOR = 0.99
        EPSILON_START = 1.0
        EPSILON_END = 0.05
        EPSILON_DECAY_STEPS = 30000
        POS_BIN_SIZE = 2.0
        AGENT_HEALTH_BINS = [0, 5, 10, 15, 20]
        GHAST_HEALTH_BINS = [0, 3, 6, 10]
        AGENT_YAW_BIN_SIZE = 45.0
        AGENT_PITCH_BINS = [-90, -45, -15, 15, 45, 90]
        MAX_FIREBALLS_TO_CONSIDER = 2
        FIREBALL_POS_BIN_SIZE = 3.0
        AGENT_START_HEALTH = 20.0
        GHAST_START_HEALTH = 10.0
        REWARD_GOT_HIT, REWARD_AGENT_DEATH, REWARD_TIME_PENALTY_STEP = -75, -150, -0.1
        REWARD_KILL_GHAST, REWARD_MISSION_SUCCESS, REWARD_HIT_GHAST_CUSTOM_BONUS, REWARD_SHOOT_ARROW = 200, 100, 10, 0


class RLAgent:
    def __init__(self, action_list, num_actions, state_size,
                 learning_rate=getattr(constants, 'DQN_LEARNING_RATE', getattr(constants, 'LEARNING_RATE', 0.001)),
                 discount_factor=getattr(constants, 'DISCOUNT_FACTOR', 0.99),
                 epsilon_start=getattr(constants, 'EPSILON_START', 1.0),
                 epsilon_end=getattr(constants, 'EPSILON_END', 0.05),
                 epsilon_decay_steps=getattr(constants, 'EPSILON_DECAY_STEPS', 30000),
                 replay_buffer_size=getattr(constants, 'REPLAY_BUFFER_SIZE', 10000),
                 batch_size=getattr(constants, 'BATCH_SIZE', 32),
                 target_network_update_freq=getattr(constants, 'TARGET_NETWORK_UPDATE_FREQUENCY', 100),
                 load_model_path=None,
                 debug=getattr(constants, 'DEBUG_AGENT', False)):

        self.action_list = action_list
        self.num_actions = num_actions
        self.state_size = state_size
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.learn_step_counter = 0 # For periodic target network update
        self.debug = debug

        # Initialize counter for epsilon decay and debug prints
        self.training_step_count = 0

        self.replay_buffer = collections.deque(maxlen=replay_buffer_size)
        
        # Clear default graph for TensorFlow 1.x
        if hasattr(tf, 'reset_default_graph'):
            tf.reset_default_graph()
        
        # Build networks
        self.q_network_output, self.state_input_q = self._build_q_network("q_network")
        self.target_q_network_output, self.state_input_target_q = self._build_q_network("target_q_network")

        # Build loss and optimizer
        with tf.variable_scope("loss_optimizer"):
            self.action_input = tf.placeholder(tf.int32, [None], name="action_input")
            self.y_input = tf.placeholder(tf.float32, [None], name="y_input")
            action_one_hot = tf.one_hot(self.action_input, self.num_actions, dtype=tf.float32)
            q_value_for_action = tf.reduce_sum(tf.multiply(self.q_network_output, action_one_hot), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.y_input - q_value_for_action))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Build target network update operations
        self.target_network_update_ops = self._build_target_update_ops("q_network", "target_q_network")

        # Initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # Load model if specified
        if load_model_path:
            self.load_model(load_model_path)
        
        # Initialize target network weights
        self.sess.run(self.target_network_update_ops)

        if self.debug:
            print("DQN Agent Initialized. StateSize: {}, Actions: {}, LR: {}".format(
                self.state_size, self.num_actions, learning_rate))

    def _build_q_network(self, scope_name):
        """Build Q-network with proper variable scoping for TensorFlow 1.x"""
        with tf.variable_scope(scope_name):
            state_input = tf.placeholder(tf.float32, [None, self.state_size], name="state_input")
            
            # First hidden layer
            with tf.variable_scope("dense1"):
                W1 = tf.get_variable("weights", [self.state_size, 128], 
                                   initializer=tf.random_normal_initializer(stddev=0.1))
                b1 = tf.get_variable("biases", [128], 
                                   initializer=tf.constant_initializer(0.0))
                dense1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
            
            # Second hidden layer
            with tf.variable_scope("dense2"):
                W2 = tf.get_variable("weights", [128, 64], 
                                   initializer=tf.random_normal_initializer(stddev=0.1))
                b2 = tf.get_variable("biases", [64], 
                                   initializer=tf.constant_initializer(0.0))
                dense2 = tf.nn.relu(tf.matmul(dense1, W2) + b2)
            
            # Output layer
            with tf.variable_scope("output"):
                W_out = tf.get_variable("weights", [64, self.num_actions], 
                                      initializer=tf.random_normal_initializer(stddev=0.1))
                b_out = tf.get_variable("biases", [self.num_actions], 
                                      initializer=tf.constant_initializer(0.0))
                output_q_values = tf.matmul(dense2, W_out) + b_out
                
            return output_q_values, state_input

    def _build_target_update_ops(self, main_scope, target_scope):
        """Build operations to copy weights from main network to target network"""
        ops = []
        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=main_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)
        
        # Sort variables by name to ensure correct pairing
        main_vars = sorted(main_vars, key=lambda v: v.name)
        target_vars = sorted(target_vars, key=lambda v: v.name)
        
        for main_var, target_var in zip(main_vars, target_vars):
            ops.append(target_var.assign(main_var))
        return ops

    def _discretize_value(self, value, bins_or_size):
        """Discretize a continuous value using bins or bin size"""
        # Handle special placeholder cases
        if isinstance(bins_or_size, str):
            if bins_or_size == "HEALTH_BINS_PLACEHOLDER_AGENT":
                bins_or_size = getattr(constants, 'AGENT_HEALTH_BINS', [0,5,10,15,20])
            elif bins_or_size == "HEALTH_BINS_PLACEHOLDER_GHAST":
                bins_or_size = getattr(constants, 'GHAST_HEALTH_BINS', [0,3,6,10])

        if isinstance(bins_or_size, list):
            # Use predefined bins
            if not bins_or_size or not all(isinstance(n, (int, float)) for n in bins_or_size):
                if self.debug: 
                    print("Warning: Invalid bins for _discretize_value: {}. Using default bin index 1.".format(bins_or_size))
                return 1 
            return int(np.digitize(value, bins_or_size))
        else:
            # Use bin size
            if not isinstance(bins_or_size, (int, float)) or bins_or_size == 0:
                if self.debug: 
                    print("Warning: Invalid bin_size for _discretize_value: {}. Using default.".format(bins_or_size))
                return int(value // 2.0) 
            return int(value // bins_or_size)

    def get_state_representation(self, malmo_obs_dict, last_known_ghast_health):
        """Convert Malmo observation to normalized state vector"""
        # Get constants with fallback values
        _POS_BIN_SIZE = getattr(constants, 'POS_BIN_SIZE', 2.0)
        _AGENT_HEALTH_BINS = getattr(constants, 'AGENT_HEALTH_BINS', [0, 5, 10, 15, 20])
        _GHAST_HEALTH_BINS = getattr(constants, 'GHAST_HEALTH_BINS', [0, 3, 6, 10])
        _AGENT_YAW_BIN_SIZE = getattr(constants, 'AGENT_YAW_BIN_SIZE', 45.0)
        _MAX_FIREBALLS_TO_CONSIDER = getattr(constants, 'MAX_FIREBALLS_TO_CONSIDER', 2)
        _FIREBALL_POS_BIN_SIZE = getattr(constants, 'FIREBALL_POS_BIN_SIZE', 3.0)
        _AGENT_START_HEALTH = getattr(constants, 'AGENT_START_HEALTH', 20.0)
        _GHAST_START_HEALTH = getattr(constants, 'GHAST_START_HEALTH', 10.0)

        # Calculate normalization factors
        max_agent_x_bin = max(1.0, float(np.ceil(15.0 / _POS_BIN_SIZE))) if _POS_BIN_SIZE > 0 else 1.0
        max_agent_z_bin = max(1.0, float(np.ceil(60.0 / _POS_BIN_SIZE))) if _POS_BIN_SIZE > 0 else 1.0
        max_agent_health_bin = max(1.0, float(len(_AGENT_HEALTH_BINS)))
        max_agent_yaw_bin = max(1.0, float(360.0 / _AGENT_YAW_BIN_SIZE)) if _AGENT_YAW_BIN_SIZE > 0 else 1.0
        max_rel_pos_ghast_bin = max(1.0, float(np.ceil(30.0 / _POS_BIN_SIZE))) if _POS_BIN_SIZE > 0 else 1.0
        max_rel_y_ghast_bin = max(1.0, float(np.ceil(20.0 / _POS_BIN_SIZE))) if _POS_BIN_SIZE > 0 else 1.0
        max_ghast_health_bin = max(1.0, float(len(_GHAST_HEALTH_BINS)))
        max_rel_pos_fireball_bin = max(1.0, float(np.ceil(15.0 / _FIREBALL_POS_BIN_SIZE))) if _FIREBALL_POS_BIN_SIZE > 0 else 1.0
        max_rel_y_fireball_bin = max(1.0, float(np.ceil(9.0 / _FIREBALL_POS_BIN_SIZE))) if _FIREBALL_POS_BIN_SIZE > 0 else 1.0

        # Handle None observation
        if malmo_obs_dict is None:
            return np.zeros((1, self.state_size), dtype=np.float32)

        # Extract agent features
        agent_x = malmo_obs_dict.get('XPos', 0)
        agent_y = malmo_obs_dict.get('YPos', 0)
        agent_z = malmo_obs_dict.get('ZPos', 0)
        agent_health_raw = malmo_obs_dict.get('Life', _AGENT_START_HEALTH)
        agent_yaw_raw = malmo_obs_dict.get('Yaw', 0)
        
        # Discretize agent features
        agent_x_binned = self._discretize_value(agent_x, _POS_BIN_SIZE)
        agent_z_binned = self._discretize_value(agent_z, _POS_BIN_SIZE)
        agent_health_binned = self._discretize_value(agent_health_raw, _AGENT_HEALTH_BINS)
        agent_yaw_binned = self._discretize_value(agent_yaw_raw % 360, _AGENT_YAW_BIN_SIZE)

        # Initialize ghast features
        ghast_present_val = 0.0
        ghast_rel_x_binned, ghast_rel_y_binned, ghast_rel_z_binned = 0, 0, 0
        ghast_health_binned = self._discretize_value(last_known_ghast_health, _GHAST_HEALTH_BINS)
        aiming_at_ghast_yaw_val, aiming_at_ghast_pitch_val = 0.0, 0.0

        # Process entities
        entities = malmo_obs_dict.get('entities', [])
        ghast_entity = None
        for entity in entities:
            if entity['name'] == 'Ghast': 
                ghast_entity = entity
                break
        
        if ghast_entity:
            ghast_present_val = 1.0
            ghast_x = ghast_entity.get('x', 0)
            ghast_y_ghast = ghast_entity.get('y', agent_y)
            ghast_z = ghast_entity.get('z', 0)
            ghast_health_raw = ghast_entity.get('life', last_known_ghast_health)
            ghast_health_binned = self._discretize_value(ghast_health_raw, _GHAST_HEALTH_BINS)
            
            # Calculate relative positions
            ghast_rel_x = ghast_x - agent_x
            ghast_rel_y = ghast_y_ghast - agent_y
            ghast_rel_z = ghast_z - agent_z
            ghast_rel_x_binned = self._discretize_value(ghast_rel_x, _POS_BIN_SIZE)
            ghast_rel_y_binned = self._discretize_value(ghast_rel_y, _POS_BIN_SIZE)
            ghast_rel_z_binned = self._discretize_value(ghast_rel_z, _POS_BIN_SIZE)

        # Process fireballs
        fireball_features_binned = []
        fireballs = [e for e in entities if e['name'] in ['Fireball', 'SmallFireball']]
        
        # Sort fireballs by distance
        if fireballs:
            fireballs.sort(key=lambda fb: np.sqrt(
                (fb.get('x', agent_x) - agent_x)**2 + 
                (fb.get('y', agent_y) - agent_y)**2 + 
                (fb.get('z', agent_z) - agent_z)**2
            ))

        for i in range(_MAX_FIREBALLS_TO_CONSIDER):
            if i < len(fireballs):
                fb = fireballs[i]
                fireball_features_binned.extend([
                    self._discretize_value(fb.get('x', agent_x) - agent_x, _FIREBALL_POS_BIN_SIZE),
                    self._discretize_value(fb.get('y', agent_y) - agent_y, _FIREBALL_POS_BIN_SIZE),
                    self._discretize_value(fb.get('z', agent_z) - agent_z, _FIREBALL_POS_BIN_SIZE)
                ])
            else:
                fireball_features_binned.extend([0, 0, 0])

        # Normalize features
        norm_agent_x = agent_x_binned / max_agent_x_bin
        norm_agent_z = agent_z_binned / max_agent_z_bin
        norm_agent_health = max(0.0, (agent_health_binned - 1.0) / max(1.0, (max_agent_health_bin - 1.0)))
        norm_agent_yaw = agent_yaw_binned / max_agent_yaw_bin
        norm_ghast_rel_x = ghast_rel_x_binned / max_rel_pos_ghast_bin
        norm_ghast_rel_y = ghast_rel_y_binned / max_rel_y_ghast_bin
        norm_ghast_rel_z = ghast_rel_z_binned / max_rel_pos_ghast_bin
        norm_ghast_health = max(0.0, (ghast_health_binned - 1.0) / max(1.0, (max_ghast_health_bin - 1.0)))

        # Normalize fireball features
        norm_fireball_features = []
        for i in range(_MAX_FIREBALLS_TO_CONSIDER):
            base_idx = i * 3
            norm_fireball_features.extend([
                fireball_features_binned[base_idx] / max_rel_pos_fireball_bin,
                fireball_features_binned[base_idx+1] / max_rel_y_fireball_bin,
                fireball_features_binned[base_idx+2] / max_rel_pos_fireball_bin,
            ])

        # Build final state vector
        state_list_normalized = [
            norm_agent_x, norm_agent_z, norm_agent_health, norm_agent_yaw,
            ghast_present_val,
            norm_ghast_rel_x, norm_ghast_rel_y, norm_ghast_rel_z, norm_ghast_health,
            aiming_at_ghast_yaw_val, aiming_at_ghast_pitch_val,
        ] + norm_fireball_features
        
        final_state_vector = np.array(state_list_normalized, dtype=np.float32)

        # Ensure correct size
        if len(final_state_vector) != self.state_size:
            if self.debug: 
                print("Warning: Normalized state length ({}) != expected state_size ({}). Adjusting.".format(
                    len(final_state_vector), self.state_size))
            adjusted_vector = np.zeros(self.state_size, dtype=np.float32)
            elements_to_copy = min(len(final_state_vector), self.state_size)
            adjusted_vector[:elements_to_copy] = final_state_vector[:elements_to_copy]
            final_state_vector = adjusted_vector
            
        # Debug output
        if self.debug and self.training_step_count > 0 and self.training_step_count % 200 == 0:
            print("Normalized State Vec: {}".format(final_state_vector))
            
        return final_state_vector.reshape(1, -1)

    def store_experience(self, state, action_index, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action_index, reward, next_state, done))

    def choose_action(self, current_state_vector):
        """Choose action using epsilon-greedy policy"""
        self.training_step_count += 1
        
        # Update epsilon with exponential decay
        if self.epsilon_decay_steps > 0:
            self.epsilon = self.epsilon_end + \
                           (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * self.training_step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.num_actions - 1)
            if self.debug and self.training_step_count % 100 == 0:
                print("Exploring (e={:.3f}): Action index {}".format(self.epsilon, action_index))
        else:
            q_values_prediction = self.sess.run(self.q_network_output, 
                                                feed_dict={self.state_input_q: current_state_vector})
            action_index = np.argmax(q_values_prediction[0])
            if self.debug and self.training_step_count % 100 == 0:
                print("Exploiting (e={:.3f}): Action index {} (Q-vals: {})".format(
                    self.epsilon, action_index, q_values_prediction[0]))
        return action_index

    def learn(self):
        """Perform one learning step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 

        # Sample random minibatch
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare batch data
        states_batch = np.array([experience[0].squeeze() for experience in minibatch]).reshape(-1, self.state_size)
        actions_batch = np.array([experience[1] for experience in minibatch])
        rewards_batch = np.array([experience[2] for experience in minibatch], dtype=np.float32)
        next_states_batch = np.array([experience[3].squeeze() for experience in minibatch]).reshape(-1, self.state_size)
        done_batch = np.array([experience[4] for experience in minibatch])

        # Get Q-values for next states from target network
        next_q_values_target_net = self.sess.run(self.target_q_network_output, 
                                                 feed_dict={self.state_input_target_q: next_states_batch})
        
        # Calculate target Q-values
        target_q_batch = []
        for i in range(self.batch_size):
            if done_batch[i]:
                target_q_batch.append(rewards_batch[i])
            else:
                target_q_batch.append(rewards_batch[i] + self.gamma * np.max(next_q_values_target_net[i]))
        
        target_q_batch = np.array(target_q_batch, dtype=np.float32)

        # Perform gradient descent
        _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict={
            self.state_input_q: states_batch,
            self.action_input: actions_batch,
            self.y_input: target_q_batch
        })

        # Update target network periodically
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_network_update_freq == 0:
            self.sess.run(self.target_network_update_ops)
            if self.debug: 
                print("INFO: Updated target network at step {}".format(self.learn_step_counter))
        
        # Debug output
        if self.debug and self.learn_step_counter % 200 == 0:
            print("Learning step {}. Loss: {:.4f}, Epsilon: {:.3f}".format(
                self.learn_step_counter, loss_val, self.epsilon))

    def calculate_custom_reward(self, current_obs_dict, prev_obs_dict,
                                prev_agent_health, current_agent_health,
                                prev_ghast_health, current_ghast_health,
                                action_command_taken, ghast_killed_flag,
                                xml_reward, time_since_mission_start_ms, mission_time_limit_ms,
                                died=False):
        """Calculate custom reward based on game state changes"""
        reward = xml_reward
        
        # Get reward constants
        _REWARD_GOT_HIT = getattr(constants, 'REWARD_GOT_HIT', -50)
        _REWARD_AGENT_DEATH = getattr(constants, 'REWARD_AGENT_DEATH', -200)
        _REWARD_TIME_PENALTY_STEP = getattr(constants, 'REWARD_TIME_PENALTY_STEP', -0.1)
        _REWARD_KILL_GHAST = getattr(constants, 'REWARD_KILL_GHAST', 200)
        _REWARD_MISSION_SUCCESS = getattr(constants, 'REWARD_MISSION_SUCCESS', 100)
        _REWARD_HIT_GHAST_CUSTOM_BONUS = getattr(constants, 'REWARD_HIT_GHAST_CUSTOM_BONUS', 10)
        _REWARD_SHOOT_ARROW = getattr(constants, 'REWARD_SHOOT_ARROW', 0)

        # Apply rewards based on state changes
        if current_agent_health < prev_agent_health and prev_agent_health > 0: 
            reward += _REWARD_GOT_HIT
            if self.debug:
                print("Agent took damage: {} -> {}".format(prev_agent_health, current_agent_health))
        
        if died: 
            reward += _REWARD_AGENT_DEATH
            if self.debug:
                print("Agent died!")
        
        reward += _REWARD_TIME_PENALTY_STEP
        
        if current_ghast_health < prev_ghast_health and prev_ghast_health > 0: 
            reward += _REWARD_HIT_GHAST_CUSTOM_BONUS
            if self.debug:
                print("Hit ghast: {} -> {}".format(prev_ghast_health, current_ghast_health))
        
        if ghast_killed_flag or (current_ghast_health <= 0 and prev_ghast_health > 0):
            reward += _REWARD_KILL_GHAST
            reward += _REWARD_MISSION_SUCCESS
            if self.debug:
                print("Ghast killed!")
        
        if action_command_taken == "attack 1": 
            reward += _REWARD_SHOOT_ARROW
            
        return reward

    def save_model(self, file_path_prefix):
        """Save the trained model"""
        try:
            save_path = self.saver.save(self.sess, file_path_prefix + ".ckpt")
            if self.debug: 
                print("Model saved in path: {}".format(save_path))
        except Exception as e:
            print("Error saving model: {}".format(e))

    def load_model(self, file_path_prefix):
        """Load a trained model"""
        try:
            self.saver.restore(self.sess, file_path_prefix + ".ckpt")
            if self.debug: 
                print("Model loaded from path: {}".format(file_path_prefix))
            self.sess.run(self.target_network_update_ops)
            if self.debug: 
                print("Target network updated after loading model.")
        except Exception as e:
            print("Error loading model (or file not found {}): {}".format(file_path_prefix, e))
            print("Starting with a new model.")
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.target_network_update_ops)

    def close_session(self):
        """Close TensorFlow session"""
        if self.sess:
            self.sess.close()
            if self.debug: 
                print("TensorFlow session closed.")
