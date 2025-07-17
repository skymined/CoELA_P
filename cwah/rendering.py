import pickle
import sys
sys.path.append("/home/sky/바탕화면/2025 MAS/CoELA_P/virtualhome/simulation")
from unity_simulator import comm_unity

#---------
# 1. Load Action Logs
#---------

with open("/home/sky/바탕화면/2025 MAS/CoELA_P/test_results/LLMs_comm_mistral-7b/logs_agent_10_put_dishwasher_0.pik", "rb") as f:
    data = pickle.load(f)

actions = data.get("action")
print("Actions loaded from script:", actions)

# 변환 시작
char0_actions = actions.get(0, [])
char1_actions = actions.get(1, [])
max_len = max(len(char0_actions), len(char1_actions))

scripts = []
for i in range(max_len):
    line_parts = []
    if i < len(char0_actions):
        line_parts.append(f"<char0> {char0_actions[i]}")
    if i < len(char1_actions):
        line_parts.append(f"<char1> {char1_actions[i]}")
    scripts.append(" | ".join(line_parts))

#---------
# 2. Load Graph(Environment)
#---------

graph_path = "/home/sky/바탕화면/2025 MAS/CoELA_P/test_results/LLMs_comm_mistral-7b/graph_agent_10_put_dishwasher_0.pik"
with open(graph_path, 'rb') as gf:
    graph = pickle.load(gf)
    task_id = 5  # task 번호를 명시하거나 동적으로 잡기

init_graph = graph[task_id]["init_graph"]
init_rooms = graph[task_id]["init_rooms"]  # ex: [11, 267]

#---------
# 3. Unity 연결 및 초기화
#---------

# Unity 실행 중인 포트에 연결 (보통 6314)
comm = comm_unity.UnityCommunication(port="6314")
comm.reset()
comm.expand_scene(init_graph)  # 초기 오브젝트 배치

# 캐릭터 초기 배치
comm.add_character("Chars/Male1", initial_room=init_rooms[0])
comm.add_character("Chars/Female1", initial_room=init_rooms[1])
print("✅ Scene initialized with characters.")

if not success_add:
    print("❌ Failed to add character")
    exit()
print("✅ Character added!")

#---------
# 4. 스크립트 실행
#---------

for script in scripts :
    success, message = comm.render_script([script],  frame_rate=15, find_solution=True, recording=True, camera_mode=["PERSON_FROM_BACK"])

if success:
    print("🎬 Script rendered successfully.")
else:
    print("❌ Script rendering failed:", message)

