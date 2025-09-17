import json
import os
from datetime import datetime
import numpy as np
import traceback
import logging
# import torch.multiprocessing as mp
# ctx = mp.get_context('spawn')
import multiprocessing as mp # todo: replace with torch.multiprocessing
from tools.utils import atomic_save, json_converter, get_bbox, get_axis_aligned_bbox

class Agent:
	def __init__(self, name, pose: list, info, sim_path, no_react=False, debug=False, logger=None):
		self.name = name
		self.pose = pose
		if info is not None:
			self.cash = info['cash']
			self.held_objects = info['held_objects']
		self.storage_path = f"{sim_path}/{name}"
		self.scratch = json.load(open(f"{self.storage_path}/scratch.json", "r"))
		self.curr_time: datetime = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
		self.seed_knowledge = json.load(open(f"{self.storage_path}/seed_knowledge.json", "r"))
		self.logger = logger
		self.no_react = no_react
		self.debug = debug
		self.WALK_SPEED = 1.0 # m/s
		self.BIKE_SPEED = 3.0 # m/s
		self.current_vehicle = None
		self.current_place = None
		self.action_status = None
		self.held_objects = [None, None]
		self.last_path = None
		self.steps = 0
		self.obs = None

	def reset(self, name, pose):
		self.name = name
		self.pose = pose
		self.curr_time = datetime.strptime(self.scratch['curr_time'], "%B %d, %Y, %H:%M:%S") if self.scratch['curr_time'] is not None else None
		self.current_vehicle = self.scratch['current_vehicle'] if 'current_vehicle' in self.scratch else None
		self.current_place = self.scratch['current_place'] if 'current_place' in self.scratch else None
		self.action_status = None

	def load_from_checkpoint(self, checkpoint):
		pass

	def act(self, obs):
		self.steps = obs['steps']
		self.pose = obs['pose']
		self.obs = obs
		if self.curr_time is None or ("start_time" in self.scratch and self.curr_time == self.scratch["start_time"]) or self.curr_time.date() != obs['curr_time'].date():
			self.logger.info(f"a new day.")
			obs['new_day'] = True
		else:
			obs['new_day'] = False
		self.curr_time = obs['curr_time']
		self.current_place = obs['current_place']
		self.current_vehicle = obs['current_vehicle']
		self.action_status = obs['action_status']
		self.held_objects = obs['held_objects']
		self.cash = obs['cash']
		self._process_obs(obs)
		if self.action_status == "ONGOING":
			return None
		action = self._act(obs)
		self.save_scratch()
		return action

	def chat(self, content):
		# override to generate the utterance when chatting
		pass

	def _process_obs(self, obs):
		# override to process observations **each turn**
		pass

	def _act(self, obs):
		# override to perform your own action **when spare**
		pass

	def save_scratch(self):
		self.scratch['held_objects'] = self.held_objects
		self.scratch['curr_time'] = self.curr_time.strftime("%B %d, %Y, %H:%M:%S")
		self.scratch['current_vehicle'] = self.current_vehicle
		self.scratch['current_place'] = self.current_place
		atomic_save(os.path.join(self.storage_path, "scratch.json"), json.dumps(self.scratch, indent=4, default=json_converter))

	def __str__(self):
		return f"Agent {self.name} at {self.pose}"

	def turn_to_pos(self, target_pos):
		cur_trans = np.array(self.pose[:2])
		target_rad = np.arctan2(target_pos[1] - cur_trans[1], target_pos[0] - cur_trans[0])
		delta_rad = target_rad - self.pose[-1]
		if delta_rad > np.pi:
			delta_rad -= 2 * np.pi
		elif delta_rad < -np.pi:
			delta_rad += 2 * np.pi
		if delta_rad > 0:
			action = {
				'type': 'turn_left',
				'arg1': np.rad2deg(delta_rad),
			}
		else:
			action = {
				'type': 'turn_right',
				'arg1': np.rad2deg(-delta_rad),
			}
		return action

	def navigate(self, sg, goal_pos, goal_bbox=None):
		if goal_pos is None:
			return None
		cur_trans = np.array(self.pose[:2])
		goal_bbox = get_bbox(goal_bbox, goal_pos)
		path = sg.volume_grid_builder.navigate(cur_trans, goal_bbox, self.last_path)
		self.last_path = None
		if path is None:
			self.logger.error(f"No path found when navigate agent {self.name} to {goal_pos}.")
			return None
		else:
			if self.current_vehicle == "bicycle":
				nav_grid_num = int(self.BIKE_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			elif self.current_vehicle is None:
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			else:
				self.logger.error(f"Unsupported vehicle type {self.current_vehicle} for navigation.")
				nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
			cur_goal = path[min(nav_grid_num, len(path) - 1)]
			if sg.volume_grid_builder.has_obstacle(get_axis_aligned_bbox(np.array([cur_goal, cur_trans]), None)):
				cur_goal = path[min(2, len(path) - 1)]
		
		self.logger.debug(f"Path {path[:3]}\n...\n{path[-3:]}")
		from .sg.builder.volume_grid import convex_hull, dist_to_hull
		dist = dist_to_hull(path[-1], convex_hull(goal_bbox))
		if dist > 2:
			self.logger.warning(f"Unable to find a path to the target bounding box. The optimal available path is still a distance of {dist} away from the target bounding box. The optimal path has been automatically adopted.")
		if self.action_status == "COLLIDE":
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is colliding with obstacles, path found was {path}.")
		# move
		target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
		delta_rad = target_rad - self.pose[-1]
		if delta_rad > np.pi:
			delta_rad -= 2 * np.pi
		elif delta_rad < -np.pi:
			delta_rad += 2 * np.pi

		if delta_rad > np.deg2rad(15):
			action = {
				'type': 'turn_left',
				'arg1': np.rad2deg(delta_rad),
			}
			self.last_path = path
		elif delta_rad < -np.deg2rad(15):
			action = {
				'type': 'turn_right',
				'arg1': np.rad2deg(-delta_rad),
			}
			self.last_path = path
		else: action = {
			'type': 'move_forward',
			'arg1': np.linalg.norm(cur_goal - cur_trans),
		}
		if action['arg1'] < 0.1:
			self.logger.warning(f"{self.name} at {self.pose} moving to {cur_goal} is too close, path found was {path}.")
		return action


class AgentProcess(mp.Process):
	def __init__(self, agent_cls: type[Agent], name: str, **kwargs: dict):
		super().__init__(daemon=True)
		self.agent_cls = agent_cls
		self.name = name
		self.logging_level: str = kwargs.pop("logging_level", "INFO")
		self.multi_process: bool = kwargs.pop("multi_process", True)
		self.sim_path: str = kwargs.get("sim_path", "curr_sim")
		self.kwargs = kwargs
		self.input_queue = mp.Queue()
		self.action_queue = mp.Queue()
		self.utterance_queue = mp.Queue()
		if not self.multi_process:
			self.agent = self.create_agent()
		else:
			self.agent = None

	def create_agent(self):
		logger = AgentLogger(self.name, self.logging_level,
							 os.path.join(self.sim_path.replace('curr_sim', 'logs'), f"{self.name}.log"))
		agent = self.agent_cls(self.name, logger=logger, **self.kwargs)
		return agent

	def log_step_agent_info(self, obs_i, action):
		obs_printable = {k: v for k, v in obs_i.items() if
						 not isinstance(v, np.ndarray) and not isinstance(v, datetime)}
		obs_printable.pop("gt_seg_entity_idx_to_info", None)
		step_info = {"curr_time": obs_i["curr_time"],
					 "obs": obs_printable,
					 "action": action,  # todo: if ongoing, then log down the last action [which is ongoing]
					 "curr_events": self.agent.curr_events if hasattr(self.agent, "curr_events") else None,
					 "react_mode": self.agent.react_mode if hasattr(self.agent, "react_mode") else None,
					 "chatting_buffer": self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None,
					 "commute_plan": self.agent.commute_plan if hasattr(self.agent, "commute_plan") else None,
					 "commute_plan_idx": self.agent.commute_plan_idx if hasattr(self.agent, "commute_plan_idx") else None, }
		if hasattr(self.agent, "curr_goal_description"):
			step_info["action_desp"] = self.agent.curr_goal_description
			step_info["action_dura"] = self.agent.curr_goal_duration
			step_info["action_location"] = self.agent.curr_goal_address
		elif hasattr(self.agent, "hourly_schedule") and self.agent.curr_schedule_idx < len(self.agent.hourly_schedule):
			step_info["action_desp"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["activity"]
			step_info["action_location"] = self.agent.hourly_schedule[self.agent.curr_schedule_idx]["place"] if "place" in \
																												self.agent.hourly_schedule[
																													self.agent.curr_schedule_idx] else None

		step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name, f"{self.agent.steps:06d}.json")
		atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))

	def run(self):
		self.agent = self.create_agent()
		self.agent.logger.info(f"Agent {self.name} started")
		while True:
			input = self.input_queue.get()
			if input["type"] == "chat":
				content = input["content"]
				try:
					utterance = self.agent.chat(content)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
					if "Stop the exp immediately" in e.args[0]:
						break # end the agent process immediately
					utterance = None
				self.utterance_queue.put(utterance)
				step_info_path = os.path.join(self.sim_path.replace('curr_sim', 'steps'), self.name,
											  f"{self.agent.steps:06d}.json")
				step_info = json.load(open(step_info_path, 'r'))
				step_info["action"]["arg1"] = utterance
				step_info["chatting_buffer"] = self.agent.chatting_buffer if hasattr(self.agent, "chatting_buffer") else None
				atomic_save(step_info_path, json.dumps(step_info, indent=2, default=json_converter))
			elif input["type"] == "obs":
				obs = input["content"]
				self.agent.logger.debug(f"Agent {self.name} received obs at {obs['curr_time']}")
				try:
					action = self.agent.act(obs)
				except Exception as e:
					self.agent.logger.critical(
						f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
					action = None
				self.action_queue.put(action)
				self.log_step_agent_info(obs, action)
			else:
				self.agent.logger.error(f"Agent {self.name} received unknown input: {input}")

	def update(self, obs):
		self.input_queue.put({"type": "obs", "content": obs})

	def act(self):
		if self.multi_process:
			return self.action_queue.get()
		else:
			obs = self.input_queue.get()["content"]
			try:
				action = self.agent.act(obs)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception: {e} with traceback: {traceback.format_exc()}")
				action = None
			self.log_step_agent_info(obs, action)
			return action

	def request_chat(self, content):
		self.input_queue.put({"type": "chat", "content": content})

	def get_utterance(self, steps):
		if self.multi_process:
			utterance = self.utterance_queue.get()
		else:
			content = self.input_queue.get()["content"]
			try:
				utterance = self.agent.chat(content)
			except Exception as e:
				self.agent.logger.error(
					f"Agent {self.name} generated an exception while generating utterance: {e} with traceback: {traceback.format_exc()}")
				utterance = None

		return utterance
	
	def close(self):
		if self.multi_process:
			if self.agent.is_alive():
				self.agent.terminate()


class AgentLogger(logging.Logger):
    def __init__(self, name: str, level: str, output_path: str = None):
        super().__init__(f"Agent <{name}>")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging._nameToLevel[level.upper()])
        self.addHandler(console_handler)
        if output_path:
            file_handler = logging.FileHandler(output_path)
            file_handler.setLevel(logging.DEBUG)
            self.addHandler(file_handler)
        self.formatter = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] [Agent <{name}>] %(message)s"
        )
        for handler in self.handlers:
            handler.setFormatter(self.formatter)