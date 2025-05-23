from __future__ import print_function

from builtins import range
from malmo import MalmoPython
import os
import sys
import time
import json

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

# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"

missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Hello world!</Summary>
              </About>
              
              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,2;1;"/>
                  
                  <DrawingDecorator>
                    <!-- clearing mission envionrment to start on blank since missions build on top of each other apparently -->
                    <DrawCuboid x1="0" y1="2" z1="0" x2="20" y2="40" z2="100" type="air"/>
                  
                    <!-- right wall -->
                    <DrawCuboid x1="0" y1="0" z1="0" x2="0" y2="20" z2="60" type="stone"/>
                    
                    <!-- left wall -->
                    <DrawCuboid x1="15" y1="0" z1="0" x2="15" y2="20" z2="60" type="stone"/>
                    
                    <!-- ceiling -->
                    <DrawCuboid x1="0" y1="20" z1="0" x2="15" y2="20" z2="60" type="stone"/>
                    
                    <!-- floor -->
                    <DrawCuboid x1="0" y1="1" z1="0" x2="15" y2="1" z2="60" type="stone"/>
                    
                    <!-- far wall -->
                    <DrawCuboid x1="0" y1="0" z1="60" x2="15" y2="20" z2="60" type="stone"/>
                    
                    <!-- close wall -->
                    <DrawCuboid x1="0" y1="0" z1="0" x2="15" y2="20" z2="0" type="stone"/>
                    
                    <!-- glowstones at z = 0 (close wall) -->
                    <DrawBlock x="4" y="9" z="0" type="glowstone"/>
                    <DrawBlock x="8" y="9" z="0" type="glowstone"/>
                    <DrawBlock x="12" y="9" z="0" type="glowstone"/>
                    
                    <!-- glowstones at z = 10 -->
                    <DrawBlock x="0" y="9" z="10" type="glowstone"/>
                    <DrawBlock x="15" y="9" z="10" type="glowstone"/>
                    <DrawBlock x="4" y="20" z="10" type="glowstone"/>
                    <DrawBlock x="8" y="20" z="10" type="glowstone"/>
                    <DrawBlock x="12" y="20" z="10" type="glowstone"/>
                    
                    <!-- glowstones at z = 20 -->
                    <DrawBlock x="0" y="9" z="20" type="glowstone"/>
                    <DrawBlock x="15" y="9" z="20" type="glowstone"/>
                    <DrawBlock x="4" y="20" z="20" type="glowstone"/>
                    <DrawBlock x="8" y="20" z="20" type="glowstone"/>
                    <DrawBlock x="12" y="20" z="20" type="glowstone"/>
                    
                    <!-- glowstones at z = 30 -->
                    <DrawBlock x="0" y="9" z="30" type="glowstone"/>
                    <DrawBlock x="15" y="9" z="30" type="glowstone"/>
                    <DrawBlock x="4" y="20" z="30" type="glowstone"/>
                    <DrawBlock x="8" y="20" z="30" type="glowstone"/>
                    <DrawBlock x="12" y="20" z="30" type="glowstone"/>
                    
                    <!-- glowstones at z = 40 -->
                    <DrawBlock x="0" y="9" z="40" type="glowstone"/>
                    <DrawBlock x="15" y="9" z="40" type="glowstone"/>
                    <DrawBlock x="4" y="20" z="40" type="glowstone"/>
                    <DrawBlock x="8" y="20" z="40" type="glowstone"/>
                    <DrawBlock x="12" y="20" z="40" type="glowstone"/>
                    
                    <!-- glowstones at z = 50 -->
                    <DrawBlock x="0" y="9" z="50" type="glowstone"/>
                    <DrawBlock x="15" y="9" z="50" type="glowstone"/>
                    <DrawBlock x="4" y="20" z="50" type="glowstone"/>
                    <DrawBlock x="8" y="20" z="50" type="glowstone"/>
                    <DrawBlock x="12" y="20" z="50" type="glowstone"/>
                    
                    <!-- glowstones at z = 60 (far wall) -->
                    <DrawBlock x="4" y="9" z="60" type="glowstone"/>
                    <DrawBlock x="8" y="9" z="60" type="glowstone"/>
                    <DrawBlock x="12" y="9" z="60" type="glowstone"/>
                    
                    <!-- ghast spawns at x="6.5" y="6" z="58" -->
                    <DrawEntity x="6.5" y="4" z="58" type="Ghast"/>
                    
                    <!-- glass barricade (to prevent Ghast from moving towards agent) -->
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
    
    if world_state:
        # print("\nInitial world_state object created. Type: {}".format(type(world_state)), end='') # Debug
        pass
    else:
        print("\nERROR: Initial world_state is None after getWorldState() call!")
        continue

    start_time_wait_loop = time.time()
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
    prev_agent_health = float(constants.AGENT_START_HEALTH if hasattr(constants, 'AGENT_START_HEALTH') else 20.0)
    prev_ghast_health = float(constants.GHAST_START_HEALTH if hasattr(constants, 'GHAST_START_HEALTH') else 10.0)
    ghast_killed_this_mission = False
    time_alive_ms = 0
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
                time_alive_ms = current_malmo_obs_dict['TimeAlive']

            current_state_tuple = agent.get_state_representation(current_malmo_obs_dict, prev_ghast_health)

            if prev_state_tuple is not None and prev_action_index is not None and prev_malmo_obs_dict is not None:
                current_agent_health = float(current_malmo_obs_dict.get('Life', prev_agent_health))
                current_ghast_entity = None
                if 'entities' in current_malmo_obs_dict:
                    for entity in current_malmo_obs_dict['entities']:
                        if entity['name'] == 'Ghast': current_ghast_entity = entity; break
                current_ghast_health = float(current_ghast_entity['life']) if current_ghast_entity else prev_ghast_health
                
                time_limit_ms_const_in_loop = constants.MISSION_TIME_LIMIT_MS 
                calculated_reward = agent.calculate_custom_reward(
                    current_malmo_obs_dict, prev_malmo_obs_dict,
                    prev_agent_health, current_agent_health,
                    prev_ghast_health, current_ghast_health,
                    ACTION_LIST[prev_action_index], ghast_killed_this_mission,
                    current_reward_from_xml, time_alive_ms, time_limit_ms_const_in_loop
                )
                episode_reward += calculated_reward
                agent.update(prev_state_tuple, prev_action_index, calculated_reward, current_state_tuple, False)

            action_index = agent.choose_action(current_state_tuple)
            action_command = ACTION_LIST[action_index]
            agent_host.sendCommand(action_command)


            prev_state_tuple = current_state_tuple
            prev_action_index = action_index
            prev_malmo_obs_dict = current_malmo_obs_dict
            if current_malmo_obs_dict:
                 prev_agent_health = float(current_malmo_obs_dict.get('Life', prev_agent_health))
            
            ghast_entity_in_obs = None
            if current_malmo_obs_dict and 'entities' in current_malmo_obs_dict:
                for entity in current_malmo_obs_dict['entities']:
                    if entity['name'] == 'Ghast': ghast_entity_in_obs = entity; break
            if ghast_entity_in_obs:
                prev_ghast_health = float(ghast_entity_in_obs['life'])
                if prev_ghast_health <= 0 and not ghast_killed_this_mission:
                    ghast_killed_this_mission = True; print("GHAST DEFEATED IN EPISODE!")
            elif prev_ghast_health > 0: pass
        
        # Corrected section for updating world_state at the end of the loop
        world_state = agent_host.getWorldState() # Get the new state for the next loop check

        if not (world_state and world_state.is_mission_running): # If mission ended, break
            break
            
        if world_state.errors: # Check for errors in the new world_state
            for error in world_state.errors: print("Error (during mission):", error.text)
        time.sleep(0.05)

    print("Episode {} finished.".format(i_episode + 1))
    print("Total reward for episode: {}".format(episode_reward))

    if mission_actually_began and prev_state_tuple is not None and prev_action_index is not None and prev_malmo_obs_dict is not None:
        time_limit_ms_const = constants.MISSION_TIME_LIMIT_MS
        final_time_for_reward = time_alive_ms

        final_calculated_reward = agent.calculate_custom_reward(
            prev_malmo_obs_dict, prev_malmo_obs_dict,
            prev_agent_health, prev_agent_health,
            prev_ghast_health, 0 if ghast_killed_this_mission else prev_ghast_health,
            ACTION_LIST[prev_action_index],
            ghast_killed_this_mission,
            current_reward_from_xml,
            final_time_for_reward,
            time_limit_ms_const,
            died=(prev_agent_health <=0)
        )
        episode_reward += final_calculated_reward
        agent.update(prev_state_tuple, prev_action_index, final_calculated_reward, None, True)
    elif not mission_actually_began:
        print("Skipping terminal reward calculation as mission did not properly begin or run.")
    else:
        print("Skipping terminal reward: No previous state/action to learn from.")

    if (i_episode + 1) % 10 == 0:
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