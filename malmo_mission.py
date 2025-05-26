from __future__ import print_function

from builtins import range
from malmo import MalmoPython
import os
import sys
import time # For manual time tracking
import json
import csv

# Attempt to import the RL agent and constants
try:
    from rl_agent import RLAgent
    import constants
except ImportError as e:
    print("ERROR: Could not import RLAgent or constants. {}".format(e))


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
else:
    import functools
    print = functools.partial(print, flush=True)

missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Ghast Battle Mission</Summary>
              </About>
              
              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,2;1;"/>
                  
                  <DrawingDecorator>
                    <DrawCuboid x1="0" y1="2" z1="-15" x2="20" y2="40" z2="100" type="air"/>
                    <DrawCuboid x1="0" y1="0" z1="-10" x2="0" y2="20" z2="60" type="stone"/>
                    <DrawCuboid x1="15" y1="0" z1="-10" x2="15" y2="20" z2="60" type="stone"/>
                    <DrawCuboid x1="0" y1="20" z1="-10" x2="15" y2="20" z2="60" type="stone"/>
                    <DrawCuboid x1="0" y1="1" z1="-10" x2="15" y2="1" z2="60" type="stone"/>
                    <DrawCuboid x1="0" y1="0" z1="60" x2="15" y2="20" z2="60" type="stone"/>
                    <DrawCuboid x1="0" y1="0" z1="-10" x2="15" y2="20" z2="-10" type="stone"/>
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
                    <DrawBlock x="4" y="9" z="-10" type="glowstone"/>
                    <DrawBlock x="8" y="9" z="-10" type="glowstone"/>
                    <DrawBlock x="12" y="9" z="-10" type="glowstone"/>
                    <DrawEntity x="6.5" y="4" z="58" type="Ghast"/>
                    <DrawCuboid x1="0" y1="0" z1="50" x2="15" y2="4" z2="50" type="glass"/>
                    <DrawCuboid x1="0" y1="8" z1="50" x2="14" y2="19" z2="59" type="glass"/>
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="60000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                  <Placement x="7" y="3" z="2" pitch="0" yaw="0"/>
                </AgentStart>
                <AgentHandlers>
                  <ObservationFromFullStats/>
                  <ContinuousMovementCommands turnSpeedDegs="180"/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

CSV_FILENAME = "ghast_battle_results.csv"

def initialize_csv(filename):
    file_exists = os.path.isfile(filename)
    try:
        is_empty = os.path.getsize(filename) == 0 if file_exists else True
    except OSError:
        is_empty = True
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or is_empty:
            writer.writerow(['Episode', 'TimeSurvived_seconds', 'FinalAgentHearts', 'GhastKilled'])

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse(sys.argv)
except RuntimeError as e:
    print('ERROR:', e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

ACTION_LIST = constants.DISCRETE_ACTIONS if hasattr(constants, 'DISCRETE_ACTIONS') else ["turn 0", "move 0"]
NUM_ACTIONS = len(ACTION_LIST)

try:
    agent = RLAgent(action_list=ACTION_LIST,
                    num_actions=NUM_ACTIONS,
                    learning_rate=constants.LEARNING_RATE,
                    discount_factor=constants.DISCOUNT_FACTOR,
                    epsilon_start=constants.EPSILON_START,
                    epsilon_end=constants.EPSILON_END,
                    epsilon_decay_steps=constants.EPSILON_DECAY_STEPS,
                    debug=constants.DEBUG_AGENT
                   )
    q_table_load_file = constants.Q_TABLE_LOAD_FILENAME if hasattr(constants, 'Q_TABLE_LOAD_FILENAME') else None
    if q_table_load_file:
        agent.load_q_table(q_table_load_file)
except AttributeError as e:
    print("Warning: Using RLAgent with default parameters due to missing constant: {}".format(e))
    agent = RLAgent(action_list=ACTION_LIST, num_actions=NUM_ACTIONS)

initialize_csv(CSV_FILENAME)

num_missions = 10 
for i_episode in range(num_missions):
    print("\n--- Starting Episode {} ---".format(i_episode + 1))
    my_mission = MalmoPython.MissionSpec(missionXML, True)
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
                print("Error starting mission after {} retries: {}".format(max_retries, e))
                mission_started_successfully = False
                break
            else:
                print("Retry {} for starting mission, error: {}".format(retry + 1, e))
                time.sleep(2)
    
    if not mission_started_successfully:
        print("Skipping episode {} due to mission start failure.".format(i_episode + 1))
        time.sleep(1)
        continue

    print("Waiting for the mission to start", end='')
    world_state = agent_host.getWorldState()
    
    if not world_state:
        print("\nERROR: Initial world_state is None after getWorldState() call!")
        continue

    episode_start_system_time_s = 0.0 # To store system time when mission truly starts
    manual_agent_survival_duration_s = 0.0 # To store manually calculated survival time if agent dies

    start_time_wait_loop = time.time() # For the "waiting for mission to begin" timeout
    max_wait_time_for_begin = 20
    mission_actually_began = False 
    while not world_state.has_mission_begun:
        print(".", end="")
        sys.stdout.flush()
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if not world_state:
            print("\nERROR: world_state became None in 'has_mission_begun' loop.")
            break 
        for error in world_state.errors:
            print("Error (while waiting for mission to begin):", error.text)
        if time.time() - start_time_wait_loop > max_wait_time_for_begin:
            print("\nTimeout waiting for mission to begin.")
            break
    
    if world_state and world_state.has_mission_begun:
        mission_actually_began = True
        episode_start_system_time_s = time.time() # MISSION TRULY STARTED: Log system start time
        print("\nMission running!")
    else:
        error_texts = [e.text for e in world_state.errors] if world_state and world_state.errors else "world_state is None or no errors"
        print("\nMission did not begin. Last known errors: {}".format(error_texts))
        time.sleep(1)
        continue
        
    episode_reward = 0
    prev_malmo_obs_dict = None
    prev_state_tuple = None
    prev_action_index = None
    initial_agent_health = float(constants.AGENT_START_HEALTH if hasattr(constants, 'AGENT_START_HEALTH') else 20.0)
    prev_agent_health = initial_agent_health
    prev_ghast_health = float(constants.GHAST_START_HEALTH if hasattr(constants, 'GHAST_START_HEALTH') else 10.0)
    ghast_killed_this_mission = False
    
    time_alive_ms_from_malmo = 0  # Stores the latest 'TimeAlive' from Malmo observations
    malmo_time_at_death_ms = 0  # Stores Malmo 'TimeAlive' when agent's health reached <=0 (for debug/comparison)
    agent_died_this_mission = False 
    current_reward_from_xml = 0.0

    while world_state and world_state.is_mission_running:
        current_malmo_obs_dict = None
        new_xml_rewards = 0.0
        if world_state.rewards:
            for r in world_state.rewards:
                new_xml_rewards += r.getValue()
        current_reward_from_xml = new_xml_rewards

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            current_malmo_obs_dict = json.loads(msg)

            if current_malmo_obs_dict is not None and 'TimeAlive' in current_malmo_obs_dict:
                time_alive_ms_from_malmo = current_malmo_obs_dict['TimeAlive']
            
            current_agent_health_from_obs = float(current_malmo_obs_dict.get('Life', prev_agent_health))
            
            if not agent_died_this_mission and current_agent_health_from_obs <= 0 and prev_agent_health > 0:
                agent_died_this_mission = True
                death_system_time_s = time.time() # Record system time at death
                manual_agent_survival_duration_s = death_system_time_s - episode_start_system_time_s
                malmo_time_at_death_ms = time_alive_ms_from_malmo # Store Malmo's TimeAlive at this point too
                
                print("DEBUG: Death Detected! Manual survival: {:.2f}s, Malmo TimeAlive at death: {}ms, Health: {}, PrevHealth: {}".format(
                    manual_agent_survival_duration_s, malmo_time_at_death_ms, current_agent_health_from_obs, prev_agent_health
                ))
            
            current_state_tuple = agent.get_state_representation(current_malmo_obs_dict, prev_ghast_health)

            if prev_state_tuple is not None and prev_action_index is not None and prev_malmo_obs_dict is not None:
                current_ghast_entity = None
                if 'entities' in current_malmo_obs_dict:
                    for entity in current_malmo_obs_dict['entities']:
                        if entity['name'] == 'Ghast': current_ghast_entity = entity; break
                current_ghast_health = float(current_ghast_entity['life']) if current_ghast_entity else prev_ghast_health
                
                time_limit_ms_const_in_loop = constants.MISSION_TIME_LIMIT_MS 
                calculated_reward = agent.calculate_custom_reward(
                    current_malmo_obs_dict, prev_malmo_obs_dict,
                    prev_agent_health, current_agent_health_from_obs, 
                    prev_ghast_health, current_ghast_health,
                    ACTION_LIST[prev_action_index], ghast_killed_this_mission,
                    current_reward_from_xml, 
                    time_alive_ms_from_malmo, 
                    time_limit_ms_const_in_loop,
                    died=agent_died_this_mission 
                )
                episode_reward += calculated_reward
                agent.update(prev_state_tuple, prev_action_index, calculated_reward, current_state_tuple, False)

            action_index = agent.choose_action(current_state_tuple)
            action_command = ACTION_LIST[action_index]
            agent_host.sendCommand(action_command)

            prev_state_tuple = current_state_tuple
            prev_action_index = action_index
            prev_malmo_obs_dict = current_malmo_obs_dict
            prev_agent_health = current_agent_health_from_obs 
            
            ghast_entity_in_obs = None
            if current_malmo_obs_dict and 'entities' in current_malmo_obs_dict:
                for entity in current_malmo_obs_dict['entities']:
                    if entity['name'] == 'Ghast': ghast_entity_in_obs = entity; break
            if ghast_entity_in_obs:
                prev_ghast_health = float(ghast_entity_in_obs['life'])
                if prev_ghast_health <= 0 and not ghast_killed_this_mission:
                    ghast_killed_this_mission = True; print("GHAST DEFEATED IN EPISODE!")
            elif not ghast_entity_in_obs and not ghast_killed_this_mission and prev_ghast_health > 0 : 
                pass 
        
        world_state = agent_host.getWorldState() 
        if not (world_state and world_state.is_mission_running): 
            break
        if world_state.errors: 
            for error in world_state.errors: print("Error (during mission):", error.text)
        time.sleep(0.05)

    final_health_for_reward_and_log = 0.0 if agent_died_this_mission else prev_agent_health

    print("Episode {} finished.".format(i_episode + 1))
    print("Total reward for episode: {:.2f}".format(episode_reward))

    # Determine time for terminal reward and logging
    time_survived_seconds_for_csv = 0.0
    time_for_terminal_reward_ms = 0.0

    if agent_died_this_mission:
        time_survived_seconds_for_csv = manual_agent_survival_duration_s
        time_for_terminal_reward_ms = manual_agent_survival_duration_s * 1000.0 # Convert manual seconds to ms
        print("Time until death (manual): {:.2f} seconds".format(manual_agent_survival_duration_s))
        print("Malmo's TimeAlive at death: {} ms".format(malmo_time_at_death_ms))
    else:
        time_survived_seconds_for_csv = time_alive_ms_from_malmo / 1000.0
        time_for_terminal_reward_ms = time_alive_ms_from_malmo # Use Malmo's total time if survived
        print("Total time active (Malmo): {:.2f} seconds".format(time_survived_seconds_for_csv))
        
    print("Final agent health (points): {:.1f}".format(final_health_for_reward_and_log))
    print("Ghast killed: {}".format(ghast_killed_this_mission))
    print("Agent died: {}".format(agent_died_this_mission))

    if prev_state_tuple is not None and prev_action_index is not None :
        time_limit_ms_const = constants.MISSION_TIME_LIMIT_MS
        final_calculated_reward = agent.calculate_custom_reward(
            prev_malmo_obs_dict if prev_malmo_obs_dict else {}, 
            prev_malmo_obs_dict if prev_malmo_obs_dict else {}, 
            prev_agent_health, 
            final_health_for_reward_and_log, 
            prev_ghast_health, 0 if ghast_killed_this_mission else prev_ghast_health,
            ACTION_LIST[prev_action_index],
            ghast_killed_this_mission,
            current_reward_from_xml, 
            time_for_terminal_reward_ms, # Use the determined time in ms
            time_limit_ms_const,
            died=agent_died_this_mission 
        )
        agent.update(prev_state_tuple, prev_action_index, final_calculated_reward, None, True)
    else:
        print("Skipping terminal Q-update: No previous state/action to learn from.")
    
    final_agent_hearts_for_csv = final_health_for_reward_and_log / 2.0

    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            i_episode + 1,
            time_survived_seconds_for_csv,
            final_agent_hearts_for_csv, 
            ghast_killed_this_mission
        ])
    print("Results for episode {} saved to {}".format(i_episode + 1, CSV_FILENAME))

    if (i_episode + 1) % 5 == 0: 
        try:
            save_filename_base = constants.Q_TABLE_SAVE_FILENAME_BASE if hasattr(constants, 'Q_TABLE_SAVE_FILENAME_BASE') else "q_table_ghast_battle"
            save_filename = "{}_episode_{}.json".format(save_filename_base, i_episode + 1)
            agent.save_q_table(save_filename)
            print("Saved Q-table after episode {}".format(i_episode+1))
        except Exception as e:
            print("Error saving Q-table: {}".format(e))
    
    time.sleep(1)

print("\n--- All Training Episodes Finished ---")
try:
    final_save_filename = constants.Q_TABLE_FINAL_SAVE_FILENAME if hasattr(constants, 'Q_TABLE_FINAL_SAVE_FILENAME') else "q_table_ghast_battle_final.json"
    agent.save_q_table(final_save_filename)
    print("Saved final Q-table.")
except Exception as e:
    print("Error saving final Q-table: {}".format(e))