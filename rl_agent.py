import random
import numpy as np
import json

# Attempt to import constants, provide defaults if not found
try:
    import constants
except ImportError:
    # This dummy class is for basic script parsing if constants.py is missing.
    # Ensure your actual constants.py is present and correct for proper operation.
    pass


class RLAgent:
    def __init__(self, action_list, num_actions,
                 learning_rate=constants.LEARNING_RATE,
                 discount_factor=constants.DISCOUNT_FACTOR,
                 epsilon_start=constants.EPSILON_START,
                 epsilon_end=constants.EPSILON_END,
                 epsilon_decay_steps=constants.EPSILON_DECAY_STEPS,
                 q_table_load_file=None,
                 debug=constants.DEBUG_AGENT if hasattr(constants, 'DEBUG_AGENT') else False):

        self.action_list = action_list
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.debug = debug

        self.q_table = {}
        self.training_step_count = 0

        if q_table_load_file:
            self.load_q_table(q_table_load_file)

        if self.debug:
            print("RLAgent Initialized. Actions: {}. LR: {}, Gamma: {}, Epsilon_start: {}".format(
                self.num_actions, self.alpha, self.gamma, self.epsilon))

    def _discretize_value(self, value, bins_or_size):


        if isinstance(bins_or_size, list):
            # Ensure bins_or_size is not empty and is a list of numbers
            if not bins_or_size or not all(isinstance(n, (int, float)) for n in bins_or_size):
                if self.debug: print("Warning: Invalid bins for _discretize_value: {}. Using default bin.".format(bins_or_size))
                return 0 # Default bin or handle error appropriately
            return int(np.digitize(value, bins_or_size))
        else:
            # Ensure bins_or_size is a non-zero number
            if not isinstance(bins_or_size, (int, float)) or bins_or_size == 0:
                if self.debug: print("Warning: Invalid bin_size for _discretize_value: {}. Using default.".format(bins_or_size))
                return int(value // 2.0) # Default bin size or handle error
            return int(value // bins_or_size)

    def get_state_representation(self, malmo_obs_dict, last_known_ghast_health):
 
        _POS_BIN_SIZE = constants.POS_BIN_SIZE 
        _AGENT_HEALTH_BINS = constants.AGENT_HEALTH_BINS 
        _GHAST_HEALTH_BINS = constants.GHAST_HEALTH_BINS 
        _AGENT_YAW_BIN_SIZE = constants.AGENT_YAW_BIN_SIZE 
        _MAX_FIREBALLS_TO_CONSIDER = constants.MAX_FIREBALLS_TO_CONSIDER 
        _FIREBALL_POS_BIN_SIZE = constants.FIREBALL_POS_BIN_SIZE 
        _AGENT_START_HEALTH = constants.AGENT_START_HEALTH 
        _GHAST_START_HEALTH = constants.GHAST_START_HEALTH 


        if malmo_obs_dict is None:
            default_agent_health = _AGENT_START_HEALTH
            default_ghast_health = _GHAST_START_HEALTH

            agent_health_bin = self._discretize_value(default_agent_health, _AGENT_HEALTH_BINS)
            ghast_health_bin = self._discretize_value(default_ghast_health, _GHAST_HEALTH_BINS)
            num_fireball_features = 3 * _MAX_FIREBALLS_TO_CONSIDER
            return (0,) * (11 + num_fireball_features)

        agent_x = malmo_obs_dict.get('XPos', 0)
        agent_y = malmo_obs_dict.get('YPos', 0)
        agent_z = malmo_obs_dict.get('ZPos', 0)
        agent_yaw = malmo_obs_dict.get('Yaw', 0)
        agent_pitch = malmo_obs_dict.get('Pitch', 0) # Not binned in example, but available
        agent_health = malmo_obs_dict.get('Life', _AGENT_START_HEALTH)


        agent_x_bin = self._discretize_value(agent_x, _POS_BIN_SIZE)
        agent_z_bin = self._discretize_value(agent_z, _POS_BIN_SIZE)
        agent_health_bin = self._discretize_value(agent_health, _AGENT_HEALTH_BINS) # Line 91
        agent_yaw_bin = self._discretize_value(agent_yaw % 360, _AGENT_YAW_BIN_SIZE)

        ghast_present = 0
        ghast_rel_x_bin, ghast_rel_y_bin, ghast_rel_z_bin = 0, 0, 0

        ghast_health_bin = self._discretize_value(last_known_ghast_health, _GHAST_HEALTH_BINS)
        aiming_at_ghast_yaw_bin = 0 # Placeholder
        aiming_at_ghast_pitch_bin = 0 # Placeholder

        entities = malmo_obs_dict.get('entities', [])
        ghast_entity = None
        for entity in entities:
            if entity['name'] == 'Ghast':
                ghast_entity = entity
                break
        
        if ghast_entity:
            ghast_present = 1
            ghast_x, ghast_y, ghast_z = ghast_entity['x'], ghast_entity['y'], ghast_entity['z']
            ghast_health = ghast_entity.get('life', last_known_ghast_health)

            ghast_health_bin = self._discretize_value(ghast_health, _GHAST_HEALTH_BINS) 

            ghast_rel_x, ghast_rel_y, ghast_rel_z = ghast_x - agent_x, ghast_y - agent_y, ghast_z - agent_z
            ghast_rel_x_bin = self._discretize_value(ghast_rel_x, _POS_BIN_SIZE)
            ghast_rel_y_bin = self._discretize_value(ghast_rel_y, _POS_BIN_SIZE) # Y-binning for Ghast
            ghast_rel_z_bin = self._discretize_value(ghast_rel_z, _POS_BIN_SIZE)
            


        fireball_features = []
        fireballs = [e for e in entities if e['name'] == 'Fireball' or e['name'] == 'SmallFireball']
        fireballs.sort(key=lambda fb: np.sqrt((fb['x']-agent_x)**2 + (fb['y']-agent_y)**2 + (fb['z']-agent_z)**2))

        for i in range(_MAX_FIREBALLS_TO_CONSIDER):
            if i < len(fireballs):
                fb = fireballs[i]
                fireball_features.extend([
                    self._discretize_value(fb['x'] - agent_x, _FIREBALL_POS_BIN_SIZE),
                    self._discretize_value(fb['y'] - agent_y, _FIREBALL_POS_BIN_SIZE),
                    self._discretize_value(fb['z'] - agent_z, _FIREBALL_POS_BIN_SIZE)
                ])
            else:
                fireball_features.extend([0, 0, 0])

        state_tuple = (
            agent_x_bin, agent_z_bin, agent_health_bin, agent_yaw_bin,
            ghast_present, ghast_rel_x_bin, ghast_rel_y_bin, ghast_rel_z_bin, ghast_health_bin,
            aiming_at_ghast_yaw_bin, aiming_at_ghast_pitch_bin,
        ) + tuple(fireball_features)

        if self.debug and self.training_step_count % 100 == 0:
            print("State: {}".format(state_tuple))
        return state_tuple

    def choose_action(self, state_tuple):
        self.training_step_count += 1
        # Epsilon decay:
        if self.epsilon_decay_steps > 0: # Avoid division by zero if decay_steps is 0
            self.epsilon = self.epsilon_end + \
                           (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * self.training_step_count / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end


        if random.random() < self.epsilon:
            action_index = random.randint(0, self.num_actions - 1)
            if self.debug and self.training_step_count % 50 == 0:
                print("Exploring (e={:.3f}): Action {}".format(self.epsilon, action_index))
        else:
            q_values = self.q_table.get(state_tuple)
            if q_values is None:
                action_index = random.randint(0, self.num_actions - 1)
                if self.debug and self.training_step_count % 50 == 0:
                    print("New state, random action: {}".format(action_index))
            else:
                action_index = np.argmax(q_values)
                if self.debug and self.training_step_count % 50 == 0:
                    print("Exploiting (e={:.3f}): Action {} from Q-vals {}".format(self.epsilon, action_index, q_values))
        return action_index

    def update(self, prev_state_tuple, action_index, reward, current_state_tuple, mission_ended):
        if prev_state_tuple is None:
            return

        current_q_values_for_prev_state = self.q_table.get(prev_state_tuple)
        if current_q_values_for_prev_state is None:
            current_q_values_for_prev_state = np.zeros(self.num_actions)
            self.q_table[prev_state_tuple] = current_q_values_for_prev_state
        
        old_q_value = current_q_values_for_prev_state[action_index]

        if mission_ended or current_state_tuple is None:
            target_q_value = reward
        else:
            future_q_values_for_current_state = self.q_table.get(current_state_tuple)
            max_future_q = 0 if future_q_values_for_current_state is None else np.max(future_q_values_for_current_state)
            target_q_value = reward + self.gamma * max_future_q
        
        new_q_value = old_q_value + self.alpha * (target_q_value - old_q_value)
        self.q_table[prev_state_tuple][action_index] = new_q_value

        if self.debug and self.training_step_count % 75 == 0:
            print("Q-Update: s={}, a={}, r={:.2f}, s'={}".format(
                prev_state_tuple, action_index, reward, current_state_tuple))
            print("  OldQ:{:.2f} -> NewQ:{:.2f} (Target:{:.2f})".format(
                old_q_value, new_q_value, target_q_value))

    def calculate_custom_reward(self, current_obs_dict, prev_obs_dict,
                                prev_agent_health, current_agent_health,
                                prev_ghast_health, current_ghast_health,
                                action_command_taken, ghast_killed_flag,
                                xml_reward, time_since_mission_start_ms, mission_time_limit_ms,
                                died=False):
        reward = xml_reward


        _REWARD_GOT_HIT = constants.REWARD_GOT_HIT 
        _REWARD_AGENT_DEATH = constants.REWARD_AGENT_DEATH 
        _REWARD_TIME_PENALTY_STEP = constants.REWARD_TIME_PENALTY_STEP 
        _REWARD_KILL_GHAST = constants.REWARD_KILL_GHAST 
        _REWARD_MISSION_SUCCESS = constants.REWARD_MISSION_SUCCESS 
        _REWARD_HIT_GHAST_CUSTOM_BONUS = constants.REWARD_HIT_GHAST_CUSTOM_BONUS 
        _REWARD_SHOOT_ARROW = constants.REWARD_SHOOT_ARROW 


        if current_agent_health < prev_agent_health:
            reward += _REWARD_GOT_HIT
            if self.debug: print("REWARD: Got hit! ({})".format(_REWARD_GOT_HIT))
        
        if died:
            reward += _REWARD_AGENT_DEATH
            if self.debug: print("REWARD: Agent DIED! ({})".format(_REWARD_AGENT_DEATH))

        reward += _REWARD_TIME_PENALTY_STEP

        if current_ghast_health < prev_ghast_health and prev_ghast_health > 0:
            reward += _REWARD_HIT_GHAST_CUSTOM_BONUS
            if self.debug: print("REWARD: Hit Ghast! (Custom Bonus: {})".format(_REWARD_HIT_GHAST_CUSTOM_BONUS))

        if ghast_killed_flag or (current_ghast_health <= 0 and prev_ghast_health > 0) :
            reward += _REWARD_KILL_GHAST
            reward += _REWARD_MISSION_SUCCESS
            if self.debug: print("REWARD: Killed Ghast! ({})".format(_REWARD_KILL_GHAST + _REWARD_MISSION_SUCCESS))
        
        # Check if shooting (even if no bow, agent might try "attack")
        action_shoot_cmd = "EXECUTE_FULL_SHOT"
        if action_command_taken == action_shoot_cmd:
             reward += _REWARD_SHOOT_ARROW # This reward is 0 if no bow, as per constants
                
        reward_looking_at_ghast = 0.0
        if current_obs_dict and 'entities' in current_obs_dict and 'XPos' in current_obs_dict and 'YPos' in current_obs_dict and 'ZPos' in obs:
            agent_x = current_obs_dict['XPos']
            agent_y = current_obs_dict['YPos']
            agent_z = current_obs_dict['ZPos']
            agent_yaw = current_obs_dict.get('Yaw', 0)
            agent_pitch = current_obs_dict.get('Pitch', 0)

            ghast_entity = next((e for e in obs['entities'] if e['name'] == 'Ghast'), None)

            if ghast_entity:
                ghast_x = ghast_entity['x']
                ghast_y = ghast_entity['y']
                ghast_z = ghast_entity['z']

                # Calculate the vector from the agent to the ghast
                vec_x = ghast_x - agent_x
                vec_y = ghast_y - agent_y
                vec_z = ghast_z - agent_z

                # Calculate the yaw and pitch needed to look directly at the ghast
                target_yaw = -math.atan2(vec_x, vec_z) * 180 / math.pi
                target_pitch = -math.atan2(vec_y, math.sqrt(vec_x**2 + vec_z**2)) * 180 / math.pi

                # Normalize yaw to be within -180 to 180 range
                agent_yaw_norm = (agent_yaw + 180) % 360 - 180
                target_yaw_norm = (target_yaw + 180) % 360 - 180

                # Check if the agent's current yaw and pitch are close to the target
                # You'll need to define a tolerance. A small tolerance means the agent
                # has to be looking very precisely at the ghast.
                YAW_TOLERANCE = 20  # degrees
                PITCH_TOLERANCE = 20 # degrees

                if (abs(agent_yaw_norm - target_yaw_norm) < YAW_TOLERANCE and
                    abs(agent_pitch - target_pitch) < PITCH_TOLERANCE):
                    reward_looking_at_ghast = constants.REWARD_LOOKING_AT_GHAST
                    if self.debug:
                        print("looking at ghast")
            
        reward += reward_looking_at_ghast


        if self.debug and self.training_step_count % 50 == 0:
            print("Calculated total reward: {:.2f}".format(reward))
        return reward

    def save_q_table(self, filename):
        try:
            serializable_q_table = {str(k): list(v) for k, v in self.q_table.items()}
            with open(filename, 'w') as f:
                json.dump(serializable_q_table, f, indent=4)
            if self.debug:
                print("Q-table saved to {}".format(filename))
        except Exception as e:
            print("Error saving Q-table: {}".format(e))

    def load_q_table(self, filename):
        try:
            with open(filename, 'r') as f:
                serializable_q_table = json.load(f)
                self.q_table = {eval(k): np.array(v) for k, v in serializable_q_table.items()}
            if self.debug:
                print("Q-table loaded from {}. Size: {} states.".format(filename, len(self.q_table)))
        except FileNotFoundError:
            if self.debug:
                print("Q-table file {} not found. Starting with an empty table.".format(filename))
        except Exception as e:
            print("Error loading Q-table: {}".format(e))