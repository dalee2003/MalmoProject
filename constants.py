# constants.py

# -----------------------------------------------------------------------------
# Action Definitions
# -----------------------------------------------------------------------------
DISCRETE_ACTIONS = [
    "move 1",           # Move forward
    "move -1",          # Move backward
    "strafe 1",         # Strafe right
    "strafe -1",        # Strafe left
    "turn 1",           # Turn right (positive value for Malmo usually means right)
    "turn -1",          # Turn left (negative value for Malmo usually means left)
    "pitch 0.1",        # Pitch down (positive for Malmo typically pitches down)
    "pitch -0.1",       # Pitch up (negative for Malmo typically pitches up)
    "attack 1",         # Start attacking
    "move 0"            # No operation / Stop current movement
]

# Index for the shooting/attack action in DISCRETE_ACTIONS.
try:
    ACTION_SHOOT_INDEX = DISCRETE_ACTIONS.index("attack 1")
except ValueError:
    ACTION_SHOOT_INDEX = -1 # Should not happen if "attack 1" is in the list
    print("WARNING: 'attack 1' not found in DISCRETE_ACTIONS for ACTION_SHOOT_INDEX.")

ACTION_STOP_ATTACK_CMD = "attack 0" # To stop continuous attack if needed
ACTION_STOP_JUMP_CMD = "jump 0"     # To ensure a single jump

# -----------------------------------------------------------------------------
# General Reinforcement Learning Hyperparameters
# -----------------------------------------------------------------------------
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 30000 # Agent steps to decay epsilon

# -----------------------------------------------------------------------------
# DQN Specific Hyperparameters
# -----------------------------------------------------------------------------
# STATE_SIZE must match the agent's get_state_representation output length
# Current agent: 4 (agent) + 1 (ghast_present) + 4 (ghast_rel) + 2 (aiming_hardcoded) + MAX_FIREBALLS_TO_CONSIDER*3
# = 4 + 1 + 4 + 2 + (2*3) = 11 + 6 = 17
STATE_SIZE = 17
DQN_LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TARGET_NETWORK_UPDATE_FREQUENCY = 200 # In terms of learning steps

# -----------------------------------------------------------------------------
# State Discretization Parameters
# -----------------------------------------------------------------------------
POS_BIN_SIZE = 2.0
AGENT_HEALTH_BINS = [0, 5, 10, 15, 20]
GHAST_HEALTH_BINS = [0, 3, 6, 10]
AGENT_YAW_BIN_SIZE = 45.0 # (360 / 45 = 8 bins)
AGENT_PITCH_BINS = [-90, -45, -15, 15, 45, 90] # For discretization if used directly

MAX_FIREBALLS_TO_CONSIDER = 2
FIREBALL_POS_BIN_SIZE = 3.0

AGENT_START_HEALTH = 20.0
GHAST_START_HEALTH = 10.0

# -----------------------------------------------------------------------------
# Reward Values
# -----------------------------------------------------------------------------
REWARD_HIT_GHAST_CUSTOM_BONUS = 10
REWARD_KILL_GHAST = 250
REWARD_SHOOT_ARROW = 0 # Agent has no bow, this is for generic attack
REWARD_MISSION_SUCCESS = 150 # Typically bundled with REWARD_KILL_GHAST by agent
REWARD_GOT_HIT = -50  # Reduced penalty
REWARD_AGENT_DEATH = -200  # Reduced penalty
REWARD_TIME_PENALTY_STEP = -0.1  # Reduced penalty
# Unused by current fixed_rl_agent's calculate_custom_reward:
# REWARD_STAY_STILL_PENALTY = -1
# REWARD_FIREBALL_NEAR_PENALTY = -5
# REWARD_WASTED_SHOT = 0

# -----------------------------------------------------------------------------
# Mission Parameters
# -----------------------------------------------------------------------------
# Note: malmo_mission.py uses getattr(constants, 'MISSION_TIME_LIMIT_MS', 30000)
# The mission XML also specifies 30000ms. Align these for clarity if needed.
MISSION_TIME_LIMIT_MS = 60000 # Increased time limit
BOW_HOTBAR_SLOT = 0

# -----------------------------------------------------------------------------
# Debugging / Control Flags
# -----------------------------------------------------------------------------
DEBUG_AGENT = True # Enables debug prints in RLAgent
DEBUG_MISSION_LOAD = False

# -----------------------------------------------------------------------------
# Model Save/Load Filenames (for DQN)
# -----------------------------------------------------------------------------
MODEL_SAVE_DIR = "saved_models" # Define a directory for models
MODEL_SAVE_FILENAME_BASE = "dqn_ghast_battle_model"
MODEL_FINAL_SAVE_FILENAME = "dqn_ghast_battle_model_final"
# To load a model, set MODEL_LOAD_FILENAME to its prefix, e.g., "saved_models/dqn_ghast_battle_model_final"
MODEL_LOAD_FILENAME = None
