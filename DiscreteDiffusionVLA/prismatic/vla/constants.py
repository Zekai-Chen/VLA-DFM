"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 3,  # SimplerEnv-Bridge: chunk=3 per concurrent DDVLA paper (Sec 4.2)
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

GOOGLE_ROBOT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,  # SimplerEnv-Fractal: chunk=8 per concurrent DDVLA paper (Sec 4.2)
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,  # base_pose_tool_reached (xyz + quat = 7) + gripper_closed (1) = 8
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

MANISKILL_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,  # Franka Panda, same as LIBERO/Fractal
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,  # tcp_pose (xyz + quat = 7) + gripper_state (1) = 8
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

CALVIN_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,  # CALVIN simulator runs at 30Hz, 8 chunks ~= 0.27s
    "ACTION_DIM": 7,         # delta xyz (3) + delta euler (3) + gripper (1)
    "PROPRIO_DIM": 15,       # CALVIN full state: ee_pos(3) + ee_orn(3) + ee_grip(1) + joints(7) + gripper_action(1)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "fractal" in cmd_args or "google_robot" in cmd_args or "rt_1" in cmd_args:
        return "GOOGLE_ROBOT"
    elif "maniskill" in cmd_args:
        return "MANISKILL"
    elif "calvin" in cmd_args:
        return "CALVIN"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    else:
        return "LIBERO"


ROBOT_PLATFORM = detect_robot_platform()

if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "GOOGLE_ROBOT":
    constants = GOOGLE_ROBOT_CONSTANTS
elif ROBOT_PLATFORM == "MANISKILL":
    constants = MANISKILL_CONSTANTS
elif ROBOT_PLATFORM == "CALVIN":
    constants = CALVIN_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `prismatic/vla/constants.py`!")
