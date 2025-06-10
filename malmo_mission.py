from __future__ import print_function

from builtins import range
import os, sys, time, json
from rl_agent import RLAgent         # just for get_state_representation & calculate_custom_reward
from dqn_agent import DQNAgent       # your new neural‐network‐based agent
import constants
import numpy as np
from malmo import MalmoPython

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

ACTION_LIST = constants.DISCRETE_ACTIONS
NUM_ACTIONS = len(ACTION_LIST)

# 1. Build a tiny RLAgent just to extract state‐size and reward logic:
state_extractor = RLAgent(action_list=ACTION_LIST,
                          num_actions=NUM_ACTIONS,
                          debug=False)

dummy_state = state_extractor.get_state_representation(None,
                                                      constants.GHAST_START_HEALTH)
state_size = len(dummy_state)

# 2. Instantiate your DQNAgent
dqn_agent = DQNAgent(state_size=state_size,
                     action_list=ACTION_LIST,
                     learning_rate=constants.DQN_LEARNING_RATE,
                     gamma=constants.DISCOUNT_FACTOR,
                     epsilon_start=constants.EPSILON_START,
                     epsilon_end=constants.EPSILON_END,
                     epsilon_decay_steps=constants.EPSILON_DECAY_STEPS,
                     debug=constants.DEBUG_AGENT)

if constants.Q_TABLE_LOAD_FILENAME: # Check if a load filename is specified in constants.py
    print("Loading DQN weights from: {}".format(constants.Q_TABLE_LOAD_FILENAME))
else:
    print("No DQN weights specified to load. Starting with new weights.")

# 3. Loop over episodes
ep_start = 215
num_missions = 5
for episode in range(ep_start, ep_start + num_missions):
    print("\n--- Episode %d ---" % episode)
    mission = MalmoPython.MissionSpec(missionXML, True)
    record = MalmoPython.MissionRecordSpec()
    # agent_host.startMission(mission, record)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent_host.startMission(mission, record)
            break
        except MalmoPython.MissionException as e:
            print("Start mission failed:", e)
            if attempt == max_retries - 1:
                raise
            time.sleep(5)

    # wait for mission to really begin …
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    # initialize tracking variables
    prev_state = None
    prev_action = None
    prev_obs = None
    prev_agent_health = constants.AGENT_START_HEALTH
    prev_ghast_health = constants.GHAST_START_HEALTH
    episode_reward = 0.0
    ghast_was_killed_during_episode = False # This flag tracks if it was killed by agent during episode

    # 4. Main step loop
    while world_state.is_mission_running:
        # pull in new observation
        if world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs = json.loads(obs_text)

            # Find Ghast entity in current observation (moved this higher to be always defined)
            gh = next((e for e in obs.get('entities', [])
                       if e['name']=='Ghast'), None)

            # build your DQN state
            curr_state = state_extractor.get_state_representation(obs,
                                                                  prev_ghast_health)

            # if we have a previous step, learn from it
            if prev_state is not None:
                # compute your custom reward
                curr_agent_health = float(obs.get('Life', prev_agent_health))
                
                # Ensure curr_ghast_health is correctly determined from obs, defaulting to previous if not found
                # 'gh' is guaranteed to be defined here now.
                curr_ghast_health = float(gh.get('life', prev_ghast_health)
                                         ) if gh else prev_ghast_health
                
                # Determine if Ghast's health went to zero in this specific step
                ghast_health_became_zero_this_step = (curr_ghast_health <= 0 and prev_ghast_health > 0)

                # Call calculate_custom_reward, now expecting two return values
                step_reward, ghast_killed_in_this_step_by_agent = state_extractor.calculate_custom_reward(
                    obs, prev_obs,
                    prev_agent_health, curr_agent_health,
                    prev_ghast_health, curr_ghast_health,
                    ACTION_LIST[prev_action],
                    ghast_killed_flag=ghast_health_became_zero_this_step, # Pass this dynamic flag
                    xml_reward=sum(r.getValue() for r in world_state.rewards),
                    time_since_mission_start_ms=obs.get('TimeAlive',0),
                    mission_time_limit_ms=constants.MISSION_TIME_LIMIT_MS,
                    died=(curr_agent_health<=0)
                )
                episode_reward += step_reward # Add the reward to total

                # If the Ghast was killed by the agent in this step, set the episode-wide flag and end the mission
                if ghast_killed_in_this_step_by_agent:
                    ghast_was_killed_during_episode = True
                    agent_host.sendCommand("quit") # Send quit command to end mission immediately
                    if constants.DEBUG_AGENT:
                        print("AGENT KILLED GHAST! Mission will now end.")

                # store in replay buffer and train
                dqn_agent.remember(prev_state, prev_action,
                                   step_reward, curr_state, False) # Use step_reward
                dqn_agent.replay()

            # choose and send next action
            action_idx = dqn_agent.act(np.array(curr_state))
            if world_state.is_mission_running:
                #add charge and shoot
                if ACTION_LIST[action_idx] == "EXECUTE_FULL_SHOT":
                    agent_host.sendCommand("use 1")     # Start charging
                    # Wait for a fixed number of game ticks.
                    # This requires your environment to tick and for you to observe those ticks.
                    # For example, if you get an observation per tick:
                    num_charge_ticks = 25  # e.g., 0.5 seconds if 20 ticks/sec
                    for _ in range(num_charge_ticks):
                        world_state = agent_host.getWorldState()
                        if not world_state.is_mission_running:
                            break
                        if world_state.number_of_observations_since_last_state > 0:
                            time.sleep(0.05)
                    if agent_host.getWorldState().is_mission_running:
                        agent_host.sendCommand("use 0") # Release/Shoot
                agent_host.sendCommand(ACTION_LIST[action_idx])

            # shift “prev” variables
            prev_state = curr_state
            prev_action = action_idx
            prev_obs = obs
            prev_agent_health = float(obs.get('Life', prev_agent_health))
            # The 'gh' variable is now guaranteed to be defined from above.
            prev_ghast_health = (gh or {'life': prev_ghast_health})['life'] # Use 'gh' directly if it was found, else use previous health

        # advance world_state
        world_state = agent_host.getWorldState()
        if world_state.errors:
            for e in world_state.errors: print("Error:", e.text)
        time.sleep(0.05)

    # 5. Terminal transition (final reward & learning)
    if prev_state is not None:
        # The ghast_killed_flag for the final reward calculation should reflect if it was killed by agent during episode.
        # Do NOT force current_ghast_health to 0.0 here for final reward.
        final_reward, _ = state_extractor.calculate_custom_reward( # Discard the second return value here
            prev_obs, prev_obs, # Pass the last observation as both current and previous for final step
            prev_agent_health, prev_agent_health, # Agent health at the end of the mission
            prev_ghast_health, prev_ghast_health, # Pass actual prev_ghast_health, not 0.0
            ACTION_LIST[prev_action],
            ghast_killed_flag=ghast_was_killed_during_episode, # Use the episode-wide flag
            xml_reward=0.0, # No additional XML rewards at end
            time_since_mission_start_ms=constants.MISSION_TIME_LIMIT_MS,
            mission_time_limit_ms=constants.MISSION_TIME_LIMIT_MS,
            died=(prev_agent_health<=0)
        )
        episode_reward += final_reward # Add the final reward to the total
        dqn_agent.remember(prev_state, prev_action,
                           final_reward, None, True)
        dqn_agent.replay()

    print("Episode %d complete, total reward = %.2f" %
          (episode, episode_reward))

    # optionally save your weights every N episodes
    if episode % 5 == 0:
        dqn_agent.model.save_weights("dqn_weights_ep%d.h5" % episode)
        print("Saved DQN weights after episode", episode)

print("All training finished.")