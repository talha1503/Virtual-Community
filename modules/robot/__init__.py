from .go2_controller import Go2Controller
from .h1_controller import H1Controller
from .google_robot_controller import GoogleRobotController
from .drone_controller import DroneController
from .husky_controller import HuskyController

ROBOT_CONTROLLERS = {
    "go2": Go2Controller,
    "h1": H1Controller,
    "google_robot": GoogleRobotController,
    "drone": DroneController,
    "husky": HuskyController
}

ROBOT_POSITION_OFFSETS = {
    "go2": (0, 0, 0.5),
    "h1": (0, 0, 1.1),
    "google_robot": (0, 0, 0.5),
    "drone": (0, 0, 0),
    "husky": (0, 0, 0.8)
}

ROBOT_CONFIGS = {
    "go2": "modules/robot/cfgs/go2/cfgs.pkl",
    "h1": "modules/robot/cfgs/h1/cfgs.pkl",
    "google_robot": "modules/robot/cfgs/google_robot/cfgs.pkl",
    "drone": "modules/robot/cfgs/drone/cfgs.pkl",
    "husky": "modules/robot/cfgs/husky/cfgs.pkl"
}
