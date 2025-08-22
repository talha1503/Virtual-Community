from .agent import Agent
from .tour_agent import TourAgent
from .logger import AgentLogger
from .go2_agent import Go2TourAgent
from .h1_agent import H1TourAgent
from .google_robot_agent import GoogleRobotTourAgent
from .husky_agent import HuskyTourAgent
from .drone_agent import DroneTourAgent

def get_agent_cls(agent_type, robot_type=None):
    if agent_type == 'tour_agent':
        if robot_type is None:
            return TourAgent
        elif robot_type == 'go2':
            return Go2TourAgent
        elif robot_type == 'h1':
            return H1TourAgent
        elif robot_type == 'google_robot':
            return GoogleRobotTourAgent
        elif robot_type == 'husky':
            return HuskyTourAgent
        elif robot_type == 'drone':
            return DroneTourAgent
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
