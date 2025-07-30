from LLM.LLM import LLM
from LLM import *

class LLM_agent:
	"""
	LLM agent class
	"""
	def __init__(self, agent_id, char_index, args):
		self.debug = args.debug
		self.agent_type = 'LLM'
		self.agent_names = ["Zero", "Alice", "Bob"]
		self.agent_id = agent_id
		self.opponent_agent_id = 3 - agent_id
		self.source = args.source
		self.lm_id = args.lm_id
		self.prompt_template_path = args.prompt_template_path
		self.communication = args.communication
		self.cot = args.cot
		self.args = args
		self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
		self.action_history = []
		self.dialogue_history = []
		self.containers_name = []
		self.goal_objects_name = []
		self.rooms_name = []
		self.roomname2id = {}
		self.unsatisfied = {}
		self.steps = 0
		# self.location = None
		# self.last_location = None
		self.plan = None
		self.stuck = 0

		# ==============이 부분 추가===============
		self.meta_plan = None  # 공유하는 장기 계획
		self.agent_role = None # 'designer' 또는 'evaluator' 역할
		self.discussion_round = 0 # 메타 플랜 논의 횟수
		self.consensus_reached = False # 계획에 대한 합의 여부
		# =============================================


		self.current_room = None
		self.last_room = None
		self.grabbed_objects = None
		self.opponent_grabbed_objects = []
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}


	@property
	def all_relative_name(self) -> list:
		return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']
	
	def goexplore(self):
		target_room_id = int(self.plan.split(' ')[-1][1:-1])
		if self.current_room['id'] == target_room_id:
			self.plan = None
			return None
		return self.plan.replace('[goexplore]', '[walktowards]')
	
	
	def gocheck(self):
		assert len(self.grabbed_objects) < 2 # must have at least one free hands
		target_container_id = int(self.plan.split(' ')[-1][1:-1])
		target_container_name = self.plan.split(' ')[1]
		target_container_room = self.id_inside_room[target_container_id]
		if self.current_room['class_name'] != target_container_room:
			return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

		target_container = self.id2node[target_container_id]
		if 'OPEN' in target_container['states']:
			self.plan = None
			return None
		if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
			return self.plan.replace('[gocheck]', '[open]') # conflict will work right?
		else:
			return self.plan.replace('[gocheck]', '[walktowards]')


	def gograb(self):
		target_object_id = int(self.plan.split(' ')[-1][1:-1])
		target_object_name = self.plan.split(' ')[1]
		if target_object_id in self.grabbed_objects:
			if self.debug:
				print(f"successful grabbed!")
			self.plan = None
			return None
		assert len(self.grabbed_objects) < 2 # must have at least one free hands

		target_object_room = self.id_inside_room[target_object_id]
		if self.current_room['class_name'] != target_object_room:
			return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

		if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
			if self.debug:
				print(f"not here any more!")
			self.plan = None
			return None
		if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
			return self.plan.replace('[gograb]', '[grab]')
		else:
			return self.plan.replace('[gograb]', '[walktowards]')
	
	def goput(self):
		# if len(self.progress['goal_location_room']) > 1: # should be ruled out
		if len(self.grabbed_objects) == 0:
			self.plan = None
			return None
		if type(self.id_inside_room[self.goal_location_id]) is list:
			if len(self.id_inside_room[self.goal_location_id]) == 0:
				print(f"never find the goal location {self.goal_location}")
				self.id_inside_room[self.goal_location_id] = self.rooms_name[:]
			target_room_name = self.id_inside_room[self.goal_location_id][0]
		else:
			target_room_name = self.id_inside_room[self.goal_location_id]

		if self.current_room['class_name'] != target_room_name:
			return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
		if self.goal_location not in self.reachable_objects:
			return f"[walktowards] {self.goal_location}"
		y = int(self.goal_location.split(' ')[-1][1:-1])
		y = self.id2node[y]
		if "CONTAINERS" in y['properties']:
			if len(self.grabbed_objects) < 2 and'CLOSED' in y['states']:
				return self.plan.replace('[goput]', '[open]')
			else:
				action = '[putin]'
		else:
			action = '[putback]'
		x = self.id2node[self.grabbed_objects[0]]
		return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


	def LLM_plan(self):
		if len(self.grabbed_objects) == 2:
			return f"[goput] {self.goal_location}", {}

		return self.LLM.run(self.current_room, [self.id2node[x] for x in self.grabbed_objects], self.satisfied, self.unchecked_containers, self.ungrabbed_objects, self.id_inside_room[self.goal_location_id], self.action_history, self.dialogue_history, self.opponent_grabbed_objects, self.id_inside_room[self.opponent_agent_id])


	def check_progress(self, state, goal_spec):
		unsatisfied = {}
		satisfied = []
		id2node = {node['id']: node for node in state['nodes']}

		for key, value in goal_spec.items():
			elements = key.split('_')
			cnt = value[0]
			for edge in state['edges']:
				if cnt == 0:
					break
				if edge['relation_type'].lower() == elements[0] and edge['to_id'] == self.goal_location_id and id2node[edge['from_id']]['class_name'] == elements[1]:
					satisfied.append(id2node[edge['from_id']])
					cnt -= 1
					# if self.debug:
					# 	print(satisfied)
			if cnt > 0:
				unsatisfied[key] = cnt
		return satisfied, unsatisfied


	def filter_graph(self, obs):
		relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
		relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
		new_graph = {
			"edges": [edge for edge in obs['edges'] if
					  edge['from_id'] in relative_id and edge['to_id'] in relative_id],
			"nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
		}
	
		return new_graph
	
	def get_action(self, observation, goal):
		"""
		:param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
		"nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
		"messages": [None, None]
		}
		:param goal:{predicate:[count, True, 2]}
		:return:
		"""
		if self.communication:
			for i in range(len(observation["messages"])):
				if observation["messages"][i] is not None:
					self.dialogue_history.append(f"{self.agent_names[i + 1]}: {observation['messages'][i]}")

		satisfied, unsatisfied = self.check_progress(observation, goal)
		# print(f"satisfied: {satisfied}")
		if len(satisfied) > 0:
			self.unsatisfied = unsatisfied
			self.satisfied = satisfied
		obs = self.filter_graph(observation)
		self.grabbed_objects = []
		opponent_grabbed_objects = []
		self.reachable_objects = []
		self.id2node = {x['id']: x for x in obs['nodes']}
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id:
				if r == 'INSIDE':
					self.current_room = self.id2node[y]
				elif r in ['HOLDS_RH', 'HOLDS_LH']:
					self.grabbed_objects.append(y)
				elif r == 'CLOSE':
					y = self.id2node[y]
					self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
			elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
				opponent_grabbed_objects.append(self.id2node[y])

		unchecked_containers = []
		ungrabbed_objects = []
		for x in obs['nodes']:
			if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
				for room, ungrabbed in self.ungrabbed_objects.items():
					if ungrabbed is None: continue
					j = None
					for i, ungrab in enumerate(ungrabbed):
						if x['id'] == ungrab['id']:
							j = i
					if j is not None:
						ungrabbed.pop(j)
				continue
			self.id_inside_room[x['id']] = self.current_room['class_name']
			if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] != self.goal_location_id:
				unchecked_containers.append(x)
			if any([x['class_name'] == g.split('_')[1] for g in self.unsatisfied]) and all([x['id'] != y['id'] for y in self.satisfied]) and 'GRABBABLE' in x['properties'] and x['id'] not in self.grabbed_objects and x['id'] not in [w['id'] for w in opponent_grabbed_objects]:
				ungrabbed_objects.append(x)

		if type(self.id_inside_room[self.goal_location_id]) is list and self.current_room['class_name'] in self.id_inside_room[self.goal_location_id]:
			self.id_inside_room[self.goal_location_id].remove(self.current_room['class_name'])
			if len(self.id_inside_room[self.goal_location_id]) == 1:
				self.id_inside_room[self.goal_location_id] = self.id_inside_room[self.goal_location_id][0]
		self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
		self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

		info = {'graph': obs,
				"obs": {
						 "grabbed_objects": self.grabbed_objects,
						 "opponent_grabbed_objects": opponent_grabbed_objects,
						 "reachable_objects": self.reachable_objects,
						 "progress": {
								"unchecked_containers": self.unchecked_containers,
								"ungrabbed_objects": self.ungrabbed_objects,
									  },
						"satisfied": self.satisfied,
						"current_room": self.current_room['class_name'],
						},
				}
		if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
			self.opponent_grabbed_objects = opponent_grabbed_objects
		action = None
		LM_times = 0
		while action is None:
			if self.plan is None:
				if LM_times > 0:
					print(info)
				if LM_times > 3:
					raise Exception(f"retrying LM_plan too many times")
				plan, a_info = self.LLM_plan()
				if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
					print("No more things to do!")
					plan = f"[wait]"
				self.plan = plan
				self.action_history.append('[send_message]' if plan.startswith('[send_message]') else plan)
				a_info.update({"steps": self.steps})
				info.update({"LLM": a_info})
				LM_times += 1
			if self.plan.startswith('[goexplore]'):
				action = self.goexplore()
			elif self.plan.startswith('[gocheck]'):
				action = self.gocheck()
			elif self.plan.startswith('[gograb]'):
				action = self.gograb()
			elif self.plan.startswith('[goput]'):
				action = self.goput()
			elif self.plan.startswith('[send_message]'):
				action = self.plan[:]
				self.plan = None
			elif self.plan.startswith('[wait]'):
				action = None
				break
			else:
				raise ValueError(f"unavailable plan {self.plan}")

		self.steps += 1
		info.update({"plan": self.plan,
					 })
		if action == self.last_action and self.current_room['class_name'] == self.last_room:
			self.stuck += 1
		else:
			self.stuck = 0
		self.last_action = action
		# self.last_location = self.location
		self.last_room = self.current_room
		if self.stuck > 20:
			print("Warning! stuck!")
			self.action_history[-1] += ' but unfinished'
			self.plan = None
			if type(self.id_inside_room[self.goal_location_id]) is list:
				target_room_name = self.id_inside_room[self.goal_location_id][0]
			else:
				target_room_name = self.id_inside_room[self.goal_location_id]
			action = f"[walktowards] {self.goal_location}"
			if self.current_room['class_name'] != target_room_name:
				action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
			self.stuck = 0
	
		return action, info

    # ============== 메타 플랜 생성하는 과정 함수 추가! ==================
    # ================================================================
	def handle_meta_plan_discussion(self, received_message=None):
		"""
        메타 플랜 생성을 위해 다중 턴 토론을 처리
		- designer : 계획을 제안/수정
		- Evaluator : 피드백 제공
		- 합의에 도달하거나 최대 토론 횟수에 도달하면 종료하기
        """
		MAX_DISCUSSION_ROUNDS = 3 # 논문에서는 3

        # 합의가 이미 이루어졌거나 토론 예산을 초과하면 더 이상 토론하지 않음
		if self.consensus_reached or self.discussion_round >= MAX_DISCUSSION_ROUNDS:
			if not self.consensus_reached: #  합의가 이루어지지 않았으면 강제 합의를 해야함
				print(f"Agent {self.agent_id}: Discussion round limit reached. Proceeding with the current meta-plan.")
				self.consensus_reached = True # 강제 합의 -> 근데 어떻게 강제 합의되는 건지 모르겠음. 그냥 강제 합의? 물어봐야겠다 ########################################확인하기##############################################
			return None 
 
		message_to_send = None # 토론을 더 진행하지 않으니까 다른 에이전트한테 보낼 메시지는 없음.

        # 1. Designer의 행동
		if self.agent_role == 'designer':
			if self.discussion_round == 0 and received_message is None: # 최초의 계획 제안
				# Cooperative Planning Module을 통해 초기 메타 플랜 생성
				self.meta_plan = self.LLM.run_initial_meta_plan(self.unsatisfied)  # 초기에 meta plan 세운 것
				print(f"Agent {self.agent_id} (Designer): Proposing initial meta-plan: {self.meta_plan}")
			else: # Evaluator로부터 피드백을 받았을 때	
				self.dialogue_history.append(f"Teammate(Evaluator): {received_message}")
				# 피드백 기반으로 메타 플랜 업데이트 (========Progress-adaptive Planning Module)
				self.meta_plan = self.LLM.run_update_meta_plan(self.meta_plan, self.dialogue_history, self.last_progress_desc)
				print(f"Agent {self.agent_id} (Designer): Proposing updated meta-plan: {self.meta_plan}")

			# Communication Module을 통해 동료에게 제안/수정안을 보내 의견 요청
			message_to_send = self.LLM.run_communication_message_designer(self.meta_plan, self.dialogue_history)
		
		# 2. Evaluator의 행동
		elif self.agent_role == 'evaluator':
			if received_message: # 평가자는 메시지를 받았을 때만 행동함!
				self.dialogue_history.append(f"Teammate(Designer): {received_message}")
				# 제안된 계획을 self.meta_plan에 임시 저장 (평가를 위해)
				proposed_plan = self.LLM.parse_plan_from_message(received_message) 
				self.meta_plan = proposed_plan
				
				# Communication Module을 통해 계획 평가 및 피드백 생성
				feedback_message = self.LLM.run_evaluate_meta_plan(self.meta_plan, self.last_progress_desc)
				
				# 피드백에서 합의 여부를 파싱하여 상태 업데이트
				if "Consensus: Yes" in feedback_message: # LLM이 합의 여부를 명시적으로 출력하니까? 자연어니까.
					self.consensus_reached = True
					print(f"Agent {self.agent_id} (Evaluator): Reached consensus on the plan.")
				else:
					print(f"Agent {self.agent_id} (Evaluator): Providing feedback: {feedback_message}")

				message_to_send = feedback_message
		
		self.discussion_round += 1 
		return message_to_send

	# ==================================================================
	# 기존은 단기 계획 생성 -> CAPO는 메타 플랩 합의 과정 거치고 sub-plan 파싱
	# ==================================================================

	def get_action(self, observation, goal):
		"""
		CAPO의 흐름에 따라 에이전트의 다음 행동을 결정
		1. 상태 업데이트 (어디에 있는지, 등등)
		2. 메타 플랜 합의
		2. LLM에게 지금 당장 해야 할 최적의 단기 행동 하나를 추천받고
		3. 추천받은 행동을 실행하고
		4. stuck 상태인지 체크하고
		5. 최종 행동 반환
		"""

		# ==================================================================
		# STEP 1: Observation 하기
		# ==================================================================
		if self.communication:
			for i in range(len(observation["messages"])):
				if observation["messages"][i] is not None:
					self.dialogue_history.append(f"{self.agent_names[i + 1]}: {observation['messages'][i]}")

		# 상태 업데이트하기 전에 완료된 목표 개수를 저장함dksl dlrdfd
		previous_satisfied_count = len(self.satisfied)
		# 새로운 관측상황을 바탕으로 현재 완료된 목표를 다시 계산함
		satisfied, unsatisfied = self.check_progress(observation, goal)

		# obs 필터링 및 에이전트/객체 상태 상세 분석 (기존 코드임))
		obs = self.filter_graph(observation) # 중요 정보만 남긴 그래프를 만드는 거라고 함
		self.id2node = {x['id']: x for x in obs['nodes']}
		self.grabbed_objects = []
		opponent_grabbed_objects = []
		self.reachable_objects = []
		
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id:
				if r == 'INSIDE': self.current_room = self.id2node[y]
				elif r in ['HOLDS_RH', 'HOLDS_LH']: self.grabbed_objects.append(y)
				elif r == 'CLOSE': self.reachable_objects.append(f"<{self.id2node[y]['class_name']}> ({y})") # 손 뻗으면 닿는물건 목록에 추가
			elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
				opponent_grabbed_objects.append(self.id2node[y])

		# info 딕셔너리 구성 (나중에 최종 반환값에 포함)
		info = {'graph': obs,
				"obs": {
						"grabbed_objects": [self.id2node[i] for i in self.grabbed_objects],
						"opponent_grabbed_objects": opponent_grabbed_objects,
						"reachable_objects": self.reachable_objects,
						"satisfied": satisfied,
						"current_room": self.current_room['class_name'], # 지금 있는 방 이름
						},
				}

		# ==================================================================
		# STEP 2: 메타 플랜 합의 단계
		# ==================================================================
		if not self.consensus_reached: # 초기가 false임
			# 1) 에이저트가 대화를 활성화한 상태인지, 2) 메시지 리스트가 존재하는지, 3) 인덱스 범위 벗어나지 않는지~ #######################3SSW : 통쨰로 문제가 있다 #
			received_message = observation["messages"][self.opponent_agent_id-1] if self.communication and observation["messages"] and len(observation["messages"]) > self.opponent_agent_id-1 else None
			# 합의가 안된 경우에 논의 시작해야 함 -> handle_meta_plan_discussion 함수 호출해서 합의한 거 보내기
			message_to_send = self.handle_meta_plan_discussion(received_message)
			
			info.update({"plan": "[discussion]", "action": message_to_send or "[wait]"})
			
			if message_to_send:
				return f"[send_message] {message_to_send}", info
			else:
				return "[wait]", info

		# ==================================================================
		# STEP 3: 진행 상황 변화에 따른 계획 재수립 단계 ########### 논문에서 New Progresss? 부분
		# ==================================================================
		if len(satisfied) > previous_satisfied_count: # 완료된 목표가 생겼다면? ####################333333333
			print(f"Agent {self.agent_id}: New progress detected! Re-evaluating the meta-plan.") #우리 계획 다시 짜야함
			self.last_progress_desc = f"Progress update: {len(satisfied)} sub-goals are completed. Task goal: {self.LLM.goal_desc}"
			self.consensus_reached = False #합의상태를 초기화 하고
			self.discussion_round = 0
			info.update({"plan": "[re-planning]", "action": "[wait]"})
			return "[wait]", info

		# ==================================================================
		# STEP 4: 합의된 메타 플랜 실행 단계
		# ==================================================================
		self.satisfied = satisfied
		self.unsatisfied = unsatisfied

		# Plan Parsing Module을 통해 현재 해야 할 sub-plan 가져오기
		# self.plan은 에이전트의 현재 sub-plan을 저장하는 멤버 변수
		self.plan = self.plan_parsing_module() # 액션 파싱하는 걸 구현하지 못함ㅇㄷㅇ##################################
		
		# 파싱된 sub-plan을 바탕으로 실제 실행할 action 생성
		action = None
		if self.plan: # 파싱된 계획이 있을 경우에만 실행
			if self.plan.startswith('[goexplore]'): action = self.goexplore()
			elif self.plan.startswith('[gocheck]'): action = self.gocheck()
			elif self.plan.startswith('[gograb]'): action = self.gograb()
			elif self.plan.startswith('[goput]'): action = self.goput()
			elif self.plan.startswith('[send_message]'): action = self.plan[:]
			elif self.plan.startswith('[wait]'): action = None
			else:
				print(f"Warning! Unknown plan parsed: {self.plan}. Waiting for next turn.")
				action = None

		# Stuck 방지 및 정보 업데이트 로직
		self.steps += 1
		
		if action == self.last_action and self.current_room['class_name'] == self.last_room:
			self.stuck += 1
		else:
			self.stuck = 0
		
		if self.stuck > 20:
			print(f"Warning! Agent {self.agent_id} is stuck with plan: {self.plan}. Resetting plan.")
			self.action_history.append(f"{self.plan} but got stuck.")
			self.plan = None
			action = None # Stuck 상태에서는 안전하게 대기
			self.stuck = 0

		self.last_action = action
		# self.last_room은 단순 문자열이 아닌 딕셔너리 객체이므로, 클래스 이름만 저장
		if self.current_room:
			self.last_room = self.current_room['class_name']

		# 최종 info 업데이트 및 반환
		info.update({"plan": self.plan, "action": action})
		return action, info


	def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal):
		self.steps = 0
		self.containers_name = containers_name
		self.goal_objects_name = goal_objects_name
		self.rooms_name = rooms_name
		self.roomname2id = {x['class_name']: x['id'] for x in room_info}
		self.id2node = {x['id']: x for x in obs['nodes']}
		self.stuck = 0
		self.last_room = None
		self.unsatisfied = {k: v[0] for k, v in goal.items()}
		self.satisfied = []
		self.goal_location = list(goal.keys())[0].split('_')[-1]
		self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
		self.id_inside_room = {self.goal_location_id: self.rooms_name[:], self.opponent_agent_id: None}
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.opponent_grabbed_objects = []
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id and r == 'INSIDE':
				self.current_room = self.id2node[y]
		self.plan = None
		self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
		self.dialogue_history = []


		# ============== 이 부분 추가함_ init에서 추가한 관련 변수를 초기화시킴 ================= 
		self.meta_plan = None
		self.agent_role = "designer" if self.agent_id == 1 else "evaluator" # 예시: Alice(id 1)가 초기 디자이너
		self.discussion_round = 0
		self.consensus_reached = False # 처음에는 합의 X
		self.last_progress_desc = "Task started"
		# =============================================

		self.LLM.reset(self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied)
