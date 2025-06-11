import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import json
import math
import csv
import os
import malmo.MalmoPython as Malmo

# Your mission XML (unchanged)
missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Hello world!</Summary>
  </About>
  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <DrawCuboid x1="0" y1="2" z1="0" x2="20" y2="40" z2="100" type="air"/>
        <DrawCuboid x1="0" y1="0" z1="0" x2="0" y2="20" z2="60" type="stone"/>
        <DrawCuboid x1="15" y1="0" z1="0" x2="15" y2="20" z2="60" type="stone"/>
        <DrawCuboid x1="0" y1="20" z1="0" x2="15" y2="20" z2="60" type="stone"/>
        <DrawCuboid x1="0" y1="1" z1="0" x2="15" y2="1" z2="60" type="stone"/>
        <DrawCuboid x1="0" y1="0" z1="60" x2="15" y2="20" z2="60" type="stone"/>
        <DrawCuboid x1="0" y1="0" z1="0" x2="15" y2="20" z2="0" type="stone"/>
        <DrawBlock x="4" y="9" z="0" type="glowstone"/>
        <DrawBlock x="8" y="9" z="0" type="glowstone"/>
        <DrawBlock x="12" y="9" z="0" type="glowstone"/>
        <DrawBlock x="0" y="9" z="10" type="glowstone"/>
        <DrawBlock x="15" y="9" z="10" type="glowstone"/>
        <DrawBlock x="4" y="20" z="10" type="glowstone"/>
        <DrawBlock x="8" y="20" z="10" type="glowstone"/>
        <DrawBlock x="12" y="20" z="10" type="glowstone"/>
        <DrawBlock x="0" y="9" z="20" type="glowstone"/>
        <DrawBlock x="15" y="9" z="20" type="glowstone"/>
        <DrawBlock x="4" y="20" z="20" type="glowstone"/>
        <DrawBlock x="8" y="20" z="20" type="glowstone"/>
        <DrawBlock x="12" y="20" z="20" type="glowstone"/>
        <DrawBlock x="0" y="9" z="30" type="glowstone"/>
        <DrawBlock x="15" y="9" z="30" type="glowstone"/>
        <DrawBlock x="4" y="20" z="30" type="glowstone"/>
        <DrawBlock x="8" y="20" z="30" type="glowstone"/>
        <DrawBlock x="12" y="20" z="30" type="glowstone"/>
        <DrawBlock x="0" y="9" z="40" type="glowstone"/>
        <DrawBlock x="15" y="9" z="40" type="glowstone"/>
        <DrawBlock x="4" y="20" z="40" type="glowstone"/>
        <DrawBlock x="8" y="20" z="40" type="glowstone"/>
        <DrawBlock x="12" y="20" z="40" type="glowstone"/>
        <DrawBlock x="0" y="9" z="50" type="glowstone"/>
        <DrawBlock x="15" y="9" z="50" type="glowstone"/>
        <DrawBlock x="4" y="20" z="50" type="glowstone"/>
        <DrawBlock x="8" y="20" z="50" type="glowstone"/>
        <DrawBlock x="12" y="20" z="50" type="glowstone"/>
        <DrawBlock x="4" y="9" z="60" type="glowstone"/>
        <DrawBlock x="8" y="9" z="60" type="glowstone"/>
        <DrawBlock x="12" y="9" z="60" type="glowstone"/>
        <DrawEntity x="6.5" y="4" z="58" type="Ghast"/>
        <DrawCuboid x1="0" y1="0" z1="50" x2="15" y2="4" z2="50" type="glass"/>
        <DrawCuboid x1="0" y1="8" z1="50" x2="14" y2="19" z2="59" type="glass"/>
        <DrawCuboid x1="3" y1="1" z1="55" x2="10" y2="1" z2="61" type="glass"/>
        <DrawCuboid x1="3" y1="8" z1="55" x2="10" y2="8" z2="61" type="glass"/>
        <DrawCuboid x1="3" y1="2" z1="61" x2="10" y2="7" z2="61" type="glass"/>
        <DrawCuboid x1="3" y1="2" z1="55" x2="3" y2="7" z2="60" type="glass"/>
        <DrawCuboid x1="10" y1="2" z1="55" x2="10" y2="7" z2="60" type="glass"/>
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="60000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  <AgentSection mode="Survival">
    <Name>MalmoTutorialBot</Name>
    <AgentStart>
      <Placement x="7" y="3" z="6" pitch="0" yaw="0"/>
      <Inventory>
        <InventoryItem slot="0" type="bow" quantity="1"/>
        <InventoryItem slot="1" type="arrow" quantity="64"/>
      </Inventory>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ObservationFromNearbyEntities>
        <Range name="entities" xrange="20" yrange="20" zrange="60"/>
      </ObservationFromNearbyEntities>
      <ContinuousMovementCommands turnSpeedDegs="180"/>
    </AgentHandlers>
  </AgentSection>
</Mission>
'''

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Action mapping (no turn/pitch, handled by aim logic)
ACTIONS = [
    {"move":1},    # Move forward
    {"move":-1},   # Move backward
    {"strafe":-1}, # Strafe left
    {"strafe":1},  # Strafe right
    {"jump":1},    # Jump
    {"pitch":0.05},  # Pitch up
    {"pitch":-0.05}, # Pitch down
    {"use":1},     # Shoot arrow (use bow)
    {},            # Do nothing
]

def get_state(obs):
    # [agent_x, agent_y, agent_z, agent_health, ghast_x, ghast_y, ghast_z, ghast_health]
    agent_x = obs.get("XPos", 0)
    agent_y = obs.get("YPos", 0)
    agent_z = obs.get("ZPos", 0)
    agent_health = obs.get("Life", 20)
    ghast_x, ghast_y, ghast_z, ghast_health = 0, 0, 0, 0
    if "entities" in obs:
        for ent in obs["entities"]:
            if ent["name"] == "Ghast":
                ghast_x = ent["x"]
                ghast_y = ent["y"]
                ghast_z = ent["z"]
                ghast_health = ent.get("life", 10)
    return np.array([[agent_x, agent_y, agent_z, agent_health, ghast_x, ghast_y, ghast_z, ghast_health]])

def get_reward(obs, prev_obs):
    reward = -1
    did_hit_ghast = False # New flag to track ghast hits
    if prev_obs is not None:
        if obs.get("Life", 20) < prev_obs.get("Life", 20):
            reward -= 10
        prev_ghast_health = 10
        ghast_health = 10
        if "entities" in prev_obs:
            for ent in prev_obs["entities"]:
                if ent["name"] == "Ghast":
                    prev_ghast_health = ent.get("life", 10)
        if "entities" in obs:
            for ent in obs["entities"]:
                if ent["name"] == "Ghast":
                    ghast_health = ent.get("life", 10)
        
        if ghast_health < prev_ghast_health:
            reward += 20
            did_hit_ghast = True # Set flag if ghast took damage
        
        if ghast_health <= 0 and prev_ghast_health > 0:
            reward += 100
            did_hit_ghast = True # Killing also counts as a hit

    return reward, did_hit_ghast # Return both reward and hit status

def is_done(obs):
    if obs is None:
        return True
    if obs.get("Life", 0) <= 0:
        return True
    if "entities" in obs:
        for ent in obs["entities"]:
            if ent["name"] == "Ghast" and ent.get("life", 10) <= 0:
                print("Ghast was killed")
                return True
    return False

def wait_for_mission_start(agent_host):
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

def get_observation(agent_host):
    world_state = agent_host.getWorldState()
    while world_state.number_of_observations_since_last_state == 0 and world_state.is_mission_running:
        time.sleep(0.05)
        world_state = agent_host.getWorldState()
    if not world_state.is_mission_running:
        return None
    obs_text = world_state.observations[-1].text
    obs = json.loads(obs_text)
    return obs

def send_action(agent_host, action):
    # Reset all movement commands
    agent_host.sendCommand("move 0")
    agent_host.sendCommand("strafe 0")
    agent_host.sendCommand("jump 0")
    agent_host.sendCommand("use 0")
    agent_host.sendCommand("pitch 0")
    # Send the selected action
    for cmd, val in action.items():
        agent_host.sendCommand("{} {}".format(cmd, val))
    if "use" in action:
        time.sleep(0.8)
        agent_host.sendCommand("use 0")

# def aim_at_ghast(agent_host, agent_obs, ghast_obs, yaw_step=2.0, pitch_step=2.0):
#     # Calculate the yaw and pitch needed to aim at the Ghast
#     x_a, y_a, z_a = agent_obs.get("XPos", 0), agent_obs.get("YPos", 0), agent_obs.get("ZPos", 0)
#     x_g, y_g, z_g = ghast_obs["x"], ghast_obs["y"], ghast_obs["z"]
#     current_yaw = agent_obs.get("Yaw", 0)
#     current_pitch = agent_obs.get("Pitch", 0)
#     dx = x_g - x_a
#     dy = y_g - y_a
#     dz = z_g - z_a
#     desired_yaw = -math.degrees(math.atan2(dx, dz))
#     desired_pitch = -math.degrees(math.atan2(dy, math.sqrt(dx**2 + dz**2)))
#     # Compute difference
#     yaw_diff = (desired_yaw - current_yaw + 180) % 360 - 180
#     pitch_diff = desired_pitch - current_pitch
#     # Send turn and pitch commands if needed
#     if abs(yaw_diff) > yaw_step:
#         agent_host.sendCommand("turn {}".format(1 if yaw_diff > 0 else -1))
#     else:
#         agent_host.sendCommand("turn 0")
#     if abs(pitch_diff) > pitch_step:
#         agent_host.sendCommand("pitch {}".format(0.05 if pitch_diff > 0 else -0.05))
#     else:
#         agent_host.sendCommand("pitch 0")
#     # If already aimed, return True
#     return abs(yaw_diff) <= yaw_step and abs(pitch_diff) <= pitch_step

def aim_at_ghast(agent_host, agent_obs, ghast_obs, yaw_step=2.0, pitch_step=2.0):
    import math
    x_a, y_a, z_a = agent_obs.get("XPos", 0), agent_obs.get("YPos", 0), agent_obs.get("ZPos", 0)
    x_g, y_g, z_g = ghast_obs["x"], ghast_obs["y"], ghast_obs["z"]
    current_yaw = agent_obs.get("Yaw", 0)
    current_pitch = agent_obs.get("Pitch", 0)
    dx = x_g - x_a
    dy = y_g - y_a
    dz = z_g - z_a
    desired_yaw = -math.degrees(math.atan2(dx, dz))
    desired_pitch = -math.degrees(math.atan2(dy, math.sqrt(dx**2 + dz**2)))
    # Compute difference
    yaw_diff = (desired_yaw - current_yaw + 180) % 360 - 180
    pitch_diff = desired_pitch - current_pitch

    # Use small discrete steps
    turn = 0
    pitch = 0
    if abs(yaw_diff) > yaw_step:
        turn = 0.1 if yaw_diff > 0 else -0.1
    if abs(pitch_diff) > pitch_step:
        pitch = 0.05 if pitch_diff > 0 else -0.05

    # Send commands for this tick only
    agent_host.sendCommand("turn {}".format(turn))
    agent_host.sendCommand("pitch {}".format(pitch))

    # Return True if aimed close enough
    return abs(yaw_diff) <= yaw_step and abs(pitch_diff) <= pitch_step

def save_rewards_to_csv(episode_rewards):
    csv_file_path = 'rewards_vs_episode.csv'
    file_exists_and_not_empty = os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0
    
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if not file_exists_and_not_empty:
            csv_writer.writerow(['Episode', 'Total Reward'])
        
        starting_episode_number = 0
        if file_exists_and_not_empty:
            with open(csv_file_path, 'r', newline='') as read_csvfile:
                csv_reader = csv.reader(read_csvfile)
                try:
                    next(csv_reader)
                    for row in csv_reader:
                        if row:
                            starting_episode_number = int(row[0])
                except StopIteration:
                    pass
                except IndexError:
                    pass

        for i, reward in enumerate(episode_rewards):
            csv_writer.writerow([starting_episode_number + i + 1, reward])

def save_shot_stats_to_csv(shot_stats_data):
    csv_file_path = 'shots_stats.csv'
    file_exists_and_not_empty = os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0

    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if not file_exists_and_not_empty:
            csv_writer.writerow(['Episode', 'Shots Hit Ghast', 'Total Shots'])
        
        starting_episode_number = 0
        if file_exists_and_not_empty:
            with open(csv_file_path, 'r', newline='') as read_csvfile:
                csv_reader = csv.reader(read_csvfile)
                try:
                    next(csv_reader)
                    for row in csv_reader:
                        if row and row[0].isdigit(): # Ensure row exists and first element is a digit
                            starting_episode_number = int(row[0])
                except StopIteration:
                    pass
                except IndexError:
                    pass

        for stat in shot_stats_data:
            # The 'stat['episode']' here is 1-indexed for the current run,
            # so we add starting_episode_number to it.
            csv_writer.writerow([starting_episode_number + stat['episode'], stat['shots_hit'], stat['total_shots']])

if __name__ == "__main__":
    state_size = 8
    action_size = len(ACTIONS)
    agent = DQNAgent(state_size, action_size)
    episodes = 2
    batch_size = 32
    episode_rewards = []
    episode_shot_stats = [] # List to store shot statistics for each episode

    agent_host = Malmo.AgentHost()

    # Client Pool Setup
    my_client_pool = Malmo.ClientPool()
    my_client_pool.add(Malmo.ClientInfo("127.0.0.1", 10000)) # Ensure client listens on this port
    experimentID = "GhastDodgeAndKillExperiment"

    for e in range(episodes):
        mission = Malmo.MissionSpec(missionXML, True)
        mission_record = Malmo.MissionRecordSpec()
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Use the client pool in startMission
                agent_host.startMission(mission, my_client_pool, mission_record, 0, experimentID)
                break
            except RuntimeError as err:
                if retry == max_retries - 1:
                    print("Error starting mission:", err)
                    exit(1)
                else:
                    time.sleep(2)
        print("Waiting for mission to start...")
        wait_for_mission_start(agent_host)
        print("Mission running")
        obs = get_observation(agent_host)
        state = get_state(obs)
        prev_obs = None
        total_reward = 0
        
        # Initialize shot counters for the current episode
        shots_taken_this_episode = 0
        shots_hit_ghast_this_episode = 0

        for t in range(1000):
            ghast = None
            if obs and "entities" in obs:
                for ent in obs["entities"]:
                    if ent["name"] == "Ghast":
                        ghast = ent
                        break
            
            aimed = True
            if ghast:
                aimed = aim_at_ghast(agent_host, obs, ghast)
            
            action_idx = agent.act(state)
            action = ACTIONS[action_idx]

            # Increment shots taken if the 'use' action (shooting bow) is chosen
            if "use" in action and action["use"] == 1:
                shots_taken_this_episode += 1
            
            send_action(agent_host, action)
            
            next_obs = get_observation(agent_host)
            if next_obs is None:
                print("Mission ended (timeout or quit).")
                break
            
            next_state = get_state(next_obs)
            
            # Get reward and check if ghast was hit
            reward, did_hit_ghast = get_reward(next_obs, obs)
            if did_hit_ghast:
                shots_hit_ghast_this_episode += 1

            done = is_done(next_obs)
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            obs = next_obs
            total_reward += reward

            if done:
                agent_host.sendCommand("quit")
                print("Done condition met (ghast killed or agent died) for episode.")
                break 
            
            agent.replay(batch_size)
            
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        print("Malmo mission confirmed terminated.")
        
        episode_rewards.append(total_reward)
        episode_shot_stats.append({
            'episode': e + 1,
            'shots_hit': shots_hit_ghast_this_episode,
            'total_shots': shots_taken_this_episode
        })
        print("Episode: {}/{}, Score: {}, Epsilon: {:.2}, Shots Hit: {}, Total Shots: {}".format(e+1, episodes, total_reward, agent.epsilon, shots_hit_ghast_this_episode, shots_taken_this_episode))
        time.sleep(1)

    print(episode_rewards) # Print total rewards list
    save_rewards_to_csv(episode_rewards)
    print("Training complete. Rewards saved to rewards_vs_episode.csv")
    
    # Save shot statistics to a new CSV file
    save_shot_stats_to_csv(episode_shot_stats)
    print("Shot statistics saved to shots_stats.csv")

