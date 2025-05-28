# constants.py

# -----------------------------------------------------------------------------
# Action Definitions
# -----------------------------------------------------------------------------
DISCRETE_ACTIONS = [
    "strafe 1",         # Strafe right
    "strafe -1",        # Strafe left
    "jump 1",           # Jump (Malmo may need "jump 0" shortly after for a single jump)

    "turn 0.02",         # Turn right (adjust sensitivity as needed)
    "turn -0.02",        # Turn left
    "pitch 0.1",        # Pitch down (look down)
    "pitch -0.1",       # Pitch up (look up)

    "attack 1",         # Start attacking (will be generic punch without bow)
    "move 0"            # No operation / Stop current movement (explicit idle)
]


# -----------------------------------------------------------------------------
# Reinforcement Learning Hyperparameters
# -----------------------------------------------------------------------------
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 30000

# -----------------------------------------------------------------------------
# State Discretization Parameters (CRUCIAL - ENSURE THESE ARE PRESENT)
# -----------------------------------------------------------------------------
POS_BIN_SIZE = 2.0                  # Bin size for X, Y, Z coordinates
AGENT_HEALTH_BINS = [0, 5, 10, 15, 20] # Bins for agent health (max 20)
GHAST_HEALTH_BINS = [0, 3, 6, 10]       # Bins for Ghast health (max 10)
AGENT_YAW_BIN_SIZE = 45.0           # Discretize yaw into 8 bins (360 / 45 = 8)
AGENT_PITCH_BINS = [-90, -45, -15, 15, 45, 90] # Example bins for pitch

MAX_FIREBALLS_TO_CONSIDER = 2       # How many nearby fireballs to put in state
FIREBALL_POS_BIN_SIZE = 3.0         # Bin size for fireball relative positions

# Default start health values (used in rl_agent.py if observation is None initially)
AGENT_START_HEALTH = 20.0
GHAST_START_HEALTH = 10.0

# -----------------------------------------------------------------------------
# Reward Values
# -----------------------------------------------------------------------------
REWARD_HIT_GHAST_CUSTOM_BONUS = 10 # If you add rewards for damaging later
REWARD_KILL_GHAST = 250
REWARD_SHOOT_ARROW = 0             # No arrows for now, so 0. Was 2.
REWARD_MISSION_SUCCESS = 150
REWARD_GOT_HIT = -100
REWARD_AGENT_DEATH = -300
REWARD_TIME_PENALTY_STEP = 0.2
REWARD_STAY_STILL_PENALTY = -1
REWARD_FIREBALL_NEAR_PENALTY = -5
REWARD_WASTED_SHOT = 0             

# -----------------------------------------------------------------------------
# Mission Parameters
# -----------------------------------------------------------------------------
MISSION_TIME_LIMIT_MS = 60000      # 60 seconds per mission/episode.
BOW_HOTBAR_SLOT = 0                # Still defined, though not used in current "no inventory" setup

# -----------------------------------------------------------------------------
# Debugging / Control Flags
# -----------------------------------------------------------------------------
DEBUG_AGENT = True
DEBUG_MISSION_LOAD = False

# -----------------------------------------------------------------------------
# Q-Table Save/Load Filenames
# -----------------------------------------------------------------------------
Q_TABLE_SAVE_FILENAME_BASE = "q_table_ghast_battle_no_inv"
Q_TABLE_FINAL_SAVE_FILENAME = "q_table_ghast_battle_no_inv_final.json"
Q_TABLE_LOAD_FILENAME = None # Set to a filename if you want to load a pre-trained table

# -----------------------------------------------------------------------------
# Deep Q Learning Parameters
# -----------------------------------------------------------------------------
BATCH_SIZE             = 32          # minibatch size for replay
MEMORY_SIZE            = 50000       # max replay buffer length
TARGET_UPDATE_FREQ     = 1000        # how many agent.train steps between target‐net sync
DQN_LEARNING_RATE      = 1e-3        # override of tabular LEARNING_RATE (if you like)
