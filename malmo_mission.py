from __future__ import print_function

from builtins import range
import MalmoPython
import os
import sys
import time

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

# More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                  
                  <ServerQuitFromTimeUp timeLimitMs="30000"/>
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



# Create default Malmo objects:

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)

# Loop until mission starts:
print("Waiting for the mission to start ", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)

print()
print("Mission running ", end=' ')

#### Moving the agent in circle so i can see what the environment looks like ####
# Turn left 90 degrees to face down the alley
# agent_host.sendCommand("turn -1")
# time.sleep(0.5)  # 180 deg/s for 0.5s = ~90 degrees
# agent_host.sendCommand("turn 0")

# Start alternating left/right movement
strafe_direction = -1  # Start by strafing left
strafe_duration = 2  # seconds per strafe direction
last_switch_time = time.time()

# Loop until mission ends:
while world_state.is_mission_running:
    current_time = time.time()

    # If time to switch strafe direction:
    if current_time - last_switch_time > strafe_duration:
        strafe_direction *= -1  # flip direction
        agent_host.sendCommand("strafe " + str(strafe_direction))
        last_switch_time = current_time

    # Keep updating world state
    world_state = agent_host.getWorldState()
    time.sleep(0.05)  # smoother loop

    for error in world_state.errors:
        print("Error:", error.text)
#### End moving agent around ####

# Stop movement after mission ends
agent_host.sendCommand("strafe 0")


print()
print("Mission ended")
# Mission has ended.w