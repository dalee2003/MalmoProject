from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warnings

from builtins import range
import tensorflow as tf
from malmo import MalmoPython
import sys
import time
import json
import numpy as np
import errno # For checking if directory exists

try:
    from rl_agent import RLAgent
    import constants
except ImportError as e:
    print("ERROR: Failed to import RLAgent or constants: {}".format(e))
    # Dummy classes (shortened for brevity, assume they are as before)
    class RLAgent: 
        dummy_init = True
        def __init__(self, **kwargs): pass
        def get_state_representation(self, obs, health): return np.zeros((1, 17))
        def choose_action(self, state): return 0
        def store_experience(self, *args): pass
        def learn(self): pass
        def calculate_custom_reward(self, *args, **kwargs): return 0
        def save_model(self, path): pass
        def close_session(self): pass
        debug = True
    class constants: 
        dummy_init = True
        DISCRETE_ACTIONS = ["move 0"]
        STATE_SIZE = 17
        MODEL_SAVE_DIR = "dummy_saved_models"
        AGENT_START_HEALTH = 20.0
        GHAST_START_HEALTH = 10.0
        ACTION_SHOOT_INDEX = -1
        ACTION_STOP_ATTACK_CMD = "attack 0"
        MISSION_TIME_LIMIT_MS = 60000
        DEBUG_AGENT = True
        MODEL_SAVE_FILENAME_BASE = "model"
        MODEL_FINAL_SAVE_FILENAME = "model_final"
        MODEL_LOAD_FILENAME = None


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools
    print = functools.partial(print, flush=True)

# Fixed mission XML - removed ServerQuitWhenAnyAgentFinishes and other potential issues
missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Ghast Battle - No Inventory, Core Observers</Summary>
    </About>
    <ServerSection>
        <ServerInitialConditions>
            <Time><StartTime>12000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
            <Weather>clear</Weather>
        </ServerInitialConditions>
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
                <DrawBlock x="4" y="9" z="0" type="glowstone"/> <DrawBlock x="8" y="9" z="0" type="glowstone"/> <DrawBlock x="12" y="9" z="0" type="glowstone"/>
                <DrawBlock x="0" y="9" z="10" type="glowstone"/> <DrawBlock x="15" y="9" z="10" type="glowstone"/>
                <DrawBlock x="4" y="20" z="10" type="glowstone"/> <DrawBlock x="8" y="20" z="10" type="glowstone"/> <DrawBlock x="12" y="20" z="10" type="glowstone"/>
                <DrawBlock x="0" y="9" z="20" type="glowstone"/> <DrawBlock x="15" y="9" z="20" type="glowstone"/>
                <DrawBlock x="4" y="20" z="20" type="glowstone"/> <DrawBlock x="8" y="20" z="20" type="glowstone"/> <DrawBlock x="12" y="20" z="20" type="glowstone"/>
                <DrawBlock x="0" y="9" z="30" type="glowstone"/> <DrawBlock x="15" y="9" z="30" type="glowstone"/>
                <DrawBlock x="4" y="20" z="30" type="glowstone"/> <DrawBlock x="8" y="20" z="30" type="glowstone"/> <DrawBlock x="12" y="20" z="30" type="glowstone"/>
                <DrawBlock x="0" y="9" z="40" type="glowstone"/> <DrawBlock x="15" y="9" z="40" type="glowstone"/>
                <DrawBlock x="4" y="20" z="40" type="glowstone"/> <DrawBlock x="8" y="20" z="40" type="glowstone"/> <DrawBlock x="12" y="20" z="40" type="glowstone"/>
                <DrawBlock x="0" y="9" z="50" type="glowstone"/> <DrawBlock x="15" y="9" z="50" type="glowstone"/>
                <DrawBlock x="4" y="20" z="50" type="glowstone"/> <DrawBlock x="8" y="20" z="50" type="glowstone"/> <DrawBlock x="12" y="20" z="50" type="glowstone"/>
                <DrawBlock x="4" y="9" z="60" type="glowstone"/> <DrawBlock x="8" y="9" z="60" type="glowstone"/> <DrawBlock x="12" y="9" z="60" type="glowstone"/>
                <DrawEntity x="6.5" y="4" z="58" type="Ghast"/>
                <DrawCuboid x1="0" y1="0" z1="50" x2="15" y2="4" z2="50" type="glass"/>
                <DrawCuboid x1="0" y1="8" z1="50" x2="14" y2="19" z2="59" type="glass"/>
            </DrawingDecorator>
            <ServerQuitFromTimeUp timeLimitMs="60000"/>
        </ServerHandlers>
    </ServerSection>
    <AgentSection mode="Survival">
        <Name>GhastFighterBot</Name>
        <AgentStart>
            <Placement x="7" y="3" z="2" pitch="0" yaw="0"/>
        </AgentStart>
        <AgentHandlers>
            <ObservationFromFullStats/>
            <DiscreteMovementCommands/>
            <ObservationFromGrid>
                <Grid name="test_minimal_grid">
                    <min x="0" y="0" z="0"/>
                    <max x="0" y="0" z="0"/>
                </Grid>
            </ObservationFromGrid>
            <ObservationFromNearbyEntities>
                <Range name="entities" xrange="100" yrange="40" zrange="100"/>
            </ObservationFromNearbyEntities>
        </AgentHandlers>
    </AgentSection>
</Mission>'''

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR parsing arguments:', e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

ACTION_LIST = getattr(constants, 'DISCRETE_ACTIONS', ["move 0", "turn 0"])
NUM_ACTIONS = len(ACTION_LIST)
STATE_SIZE = getattr(constants, 'STATE_SIZE', 17)
MODEL_SAVE_DIR = getattr(constants, 'MODEL_SAVE_DIR', "saved_models")

# Ensure the model save directory exists
try:
    os.makedirs(MODEL_SAVE_DIR)
    print("INFO: Created directory for saving models: {}".format(MODEL_SAVE_DIR))
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
    pass

try:
    # Construct full path for loading if MODEL_LOAD_FILENAME is set
    load_model_full_path = None
    _model_load_filename = getattr(constants, 'MODEL_LOAD_FILENAME', None)
    if _model_load_filename:
        load_model_full_path = os.path.join(MODEL_SAVE_DIR, _model_load_filename)
        print("INFO: Attempting to load model from: {}".format(load_model_full_path))

    agent = RLAgent(action_list=ACTION_LIST,
                    num_actions=NUM_ACTIONS,
                    state_size=STATE_SIZE,
                    learning_rate=getattr(constants, 'DQN_LEARNING_RATE', 0.001),
                    discount_factor=getattr(constants, 'DISCOUNT_FACTOR', 0.99),
                    epsilon_start=getattr(constants, 'EPSILON_START', 1.0),
                    epsilon_end=getattr(constants, 'EPSILON_END', 0.05),
                    epsilon_decay_steps=getattr(constants, 'EPSILON_DECAY_STEPS', 30000),
                    replay_buffer_size=getattr(constants, 'REPLAY_BUFFER_SIZE', 50000),
                    batch_size=getattr(constants, 'BATCH_SIZE', 64),
                    target_network_update_freq=getattr(constants, 'TARGET_NETWORK_UPDATE_FREQUENCY', 200),
                    load_model_path=load_model_full_path,
                    debug=getattr(constants, 'DEBUG_AGENT', True)
                   )
except Exception as e:
    print("CRITICAL ERROR during RLAgent initialization: {}".format(e))
    import traceback
    traceback.print_exc()
    exit(1)

print("Available actions: {}".format(ACTION_LIST))

num_missions = 10
for i_episode in range(num_missions):
    print("\n--- Starting Episode: {} ---".format(i_episode + 1))
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    mission_time_limit_seconds = getattr(constants, 'MISSION_TIME_LIMIT_MS', 60000) / 1000.0
    my_mission.timeLimitInSeconds(mission_time_limit_seconds)
    my_mission_record = MalmoPython.MissionRecordSpec()

    max_retries = 3
    mission_started_successfully = False
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_mission_record)
            mission_started_successfully = True
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("FATAL: Error starting mission after {} retries: {}".format(max_retries, e))
                break
            else:
                print("Retrying mission start (attempt {}/{}): Error: {}".format(retry + 2, max_retries, e))
                time.sleep(2.5)

    if not mission_started_successfully:
        print("Episode {} skipped: mission could not be started.".format(i_episode + 1))
        time.sleep(1)
        continue

    print("Waiting for mission to begin", end='')
    world_state = agent_host.getWorldState()

    if not world_state:
        print("\nCRITICAL ERROR: Initial world_state is None after startMission. Skipping episode.")
        continue

    wait_start_time = time.time()
    max_wait_seconds = 20
    while not world_state.has_mission_begun:
        print(".", end="")
        sys.stdout.flush()
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if not world_state:
            print("\nCRITICAL ERROR: world_state became None while waiting for mission to begin. Skipping episode.")
            break
        for error in world_state.errors:
            print("\nMalmo Error (waiting for start): {}".format(error.text))
        if time.time() - wait_start_time > max_wait_seconds:
            print("\nTIMEOUT: Mission did not begin within {} seconds.".format(max_wait_seconds))
            break

    if not (world_state and world_state.has_mission_begun):
        error_texts = [e.text for e in world_state.errors] if world_state and world_state.errors else "No specific errors, or world_state invalid."
        print("\nEpisode {} aborted: Mission did not properly begin. Errors: {}".format(i_episode + 1, error_texts))
        time.sleep(1)
        continue
    else:
        print("\nMission running!")

    episode_reward = 0.0
    prev_malmo_obs_dict = None
    prev_state_tuple_for_replay = None

    prev_agent_health = float(getattr(constants, 'AGENT_START_HEALTH', 20.0))
    prev_ghast_health = float(getattr(constants, 'GHAST_START_HEALTH', 10.0))
    ghast_killed_this_mission = False
    time_alive_ms = 0

    # Wait for first observation
    observation_wait_start = time.time()
    while world_state.number_of_observations_since_last_state == 0:
        if time.time() - observation_wait_start > 2.0:  # Wait max 2 seconds
            break
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if not world_state.is_mission_running:
            break

    # Initialize state properly
    if world_state and world_state.number_of_observations_since_last_state > 0:
        initial_msg = world_state.observations[-1].text
        try:
            initial_obs_dict = json.loads(initial_msg)
            prev_malmo_obs_dict = initial_obs_dict
            if 'Life' in initial_obs_dict: 
                prev_agent_health = float(initial_obs_dict['Life'])
            initial_ghast_entity = next((e for e in initial_obs_dict.get('entities', []) if e['name'] == 'Ghast'), None)
            if initial_ghast_entity and 'life' in initial_ghast_entity: 
                prev_ghast_health = float(initial_ghast_entity['life'])
            prev_state_tuple_for_replay = agent.get_state_representation(initial_obs_dict, prev_ghast_health)
            if 'TimeAlive' in initial_obs_dict: 
                time_alive_ms = initial_obs_dict['TimeAlive']
            print("INFO: Initial observation processed. Agent health: {}, Ghast health: {}".format(prev_agent_health, prev_ghast_health))
        except json.JSONDecodeError:
            print("ERROR: Could not decode initial JSON observation. Using default state.")
            prev_malmo_obs_dict = None
            prev_state_tuple_for_replay = agent.get_state_representation(None, prev_ghast_health)
    else:
        print("INFO: No initial observation available. Using default state.")
        prev_malmo_obs_dict = None
        prev_state_tuple_for_replay = agent.get_state_representation(None, prev_ghast_health)

    stepcounter = 0
    
    # Main mission loop
    while world_state and world_state.is_mission_running:
        stepcounter += 1
        
        # Debug output
        if stepcounter % 10 == 0:
            print("--- Episode {}, Step {} ---".format(i_episode + 1, stepcounter))
            print("Agent health: {}, Ghast health: {}, Mission running: {}".format(
                prev_agent_health, prev_ghast_health, world_state.is_mission_running))

        # Choose and execute action
        action_index = agent.choose_action(prev_state_tuple_for_replay)
        action_command = ACTION_LIST[action_index]
        
        print("Step {}: Executing action '{}' (index {})".format(stepcounter, action_command, action_index))
        agent_host.sendCommand(action_command)

        # Handle attack commands
        action_shoot_idx = getattr(constants, 'ACTION_SHOOT_INDEX', -1)
        if action_shoot_idx != -1 and action_index == action_shoot_idx:
            time.sleep(0.05)
            agent_host.sendCommand(getattr(constants, 'ACTION_STOP_ATTACK_CMD', "attack 0"))

        # Wait for world to update
        time.sleep(0.2)  # Increased wait time
        
        # Get updated world state
        world_state = agent_host.getWorldState()
        
        # Check for errors
        if world_state and world_state.errors:
            for error in world_state.errors: 
                print("Malmo Error: {}".format(error.text))

        # Debug mission status
        if world_state:
            print("Step {}: Mission running: {}, Has observations: {}".format(
                stepcounter, world_state.is_mission_running, world_state.number_of_observations_since_last_state > 0))
        else:
            print("Step {}: World state is None!".format(stepcounter))
            break

        # Process observations
        current_malmo_obs_dict = None
        current_agent_health = prev_agent_health
        current_ghast_health = prev_ghast_health
        
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            try:
                current_malmo_obs_dict = json.loads(msg)
                if 'TimeAlive' in current_malmo_obs_dict: 
                    time_alive_ms = current_malmo_obs_dict['TimeAlive']
                current_agent_health = float(current_malmo_obs_dict.get('Life', prev_agent_health))
                
                # Update ghast health
                current_ghast_entity = next((e for e in current_malmo_obs_dict.get('entities',[]) if e['name'] == 'Ghast'), None)
                if current_ghast_entity and 'life' in current_ghast_entity: 
                    current_ghast_health = float(current_ghast_entity['life'])
                elif not current_ghast_entity and prev_ghast_health > 0: 
                    current_ghast_health = 0
                    
                print("Step {}: Agent health: {}, Ghast health: {}".format(
                    stepcounter, current_agent_health, current_ghast_health))
                    
            except json.JSONDecodeError:
                print("ERROR: Could not decode JSON observation")
                current_malmo_obs_dict = prev_malmo_obs_dict

        # Check if ghast was killed
        ghast_just_killed = (current_ghast_health <= 0 and prev_ghast_health > 0)
        if ghast_just_killed and not ghast_killed_this_mission:
            ghast_killed_this_mission = True
            print("INFO: Ghast killed!")

        # Check if agent died
        agent_died = (current_agent_health <= 0)
        if agent_died:
            print("INFO: Agent died!")

        # Get current state representation
        current_state_tuple = agent.get_state_representation(current_malmo_obs_dict, current_ghast_health)

        # Calculate reward
        step_xml_rewards = 0.0
        if world_state.rewards:
            for r in world_state.rewards: 
                step_xml_rewards += r.getValue()

        calculated_reward = agent.calculate_custom_reward(
            current_malmo_obs_dict, prev_malmo_obs_dict,
            prev_agent_health, current_agent_health,
            prev_ghast_health, current_ghast_health,
            action_command, ghast_killed_this_mission,
            step_xml_rewards, time_alive_ms,
            getattr(constants, 'MISSION_TIME_LIMIT_MS', 60000),
            died=agent_died
        )
        episode_reward += calculated_reward

        # Store experience and learn
        is_done = not world_state.is_mission_running
        agent.store_experience(prev_state_tuple_for_replay, action_index, calculated_reward, current_state_tuple, is_done)
        agent.learn()

        # Update for next iteration
        prev_agent_health = current_agent_health
        prev_ghast_health = current_ghast_health
        prev_state_tuple_for_replay = current_state_tuple
        if current_malmo_obs_dict is not None: 
            prev_malmo_obs_dict = current_malmo_obs_dict

        # Check termination conditions
        
        if not world_state.is_mission_running:
            print("This is the FALALALALLALAMission ended naturally after {} steps".format(stepcounter))
            break
        
        if agent_died:
            print("Ending mission due to agent death")
            break
            
        if ghast_killed_this_mission:
            print("Ending mission due to ghast death")
            break

        # Safety check - if too many steps, break
        if stepcounter >= 1000:
            print("Ending mission due to step limit")
            break
            
        print("ended loop")

    print("Episode {} ended after {} steps. Total reward: {:.2f}. Ghast killed: {}, Agent died: {}".format(
        i_episode + 1, stepcounter, episode_reward, ghast_killed_this_mission, current_agent_health <= 0))

    # Save model periodically
    if (i_episode + 1) % 10 == 0:
        try:
            save_filename_base = getattr(constants, 'MODEL_SAVE_FILENAME_BASE', "model_data")
            save_filename_prefix = os.path.join(MODEL_SAVE_DIR, "{}_episode_{}".format(save_filename_base, i_episode + 1))
            if hasattr(agent, 'save_model'):
                agent.save_model(save_filename_prefix)
        except Exception as e:
            print("ERROR saving model checkpoint: {}".format(e))
    
    time.sleep(1)  # Pause between episodes

print("\n--- Training Complete ---")
try:
    final_save_filename_base = getattr(constants, 'MODEL_FINAL_SAVE_FILENAME', "model_data_final")
    final_save_path_prefix = os.path.join(MODEL_SAVE_DIR, final_save_filename_base)
    if hasattr(agent, 'save_model'):
        agent.save_model(final_save_path_prefix)
except Exception as e:
    print("ERROR saving final model: {}".format(e))

if hasattr(agent, 'close_session'):
    agent.close_session()

print("Mission script finished.")
