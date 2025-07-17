kill -9 $(lsof -t -i :6314)

python testing_agents/test_symbolic_LLMs.py \
--communication \
--prompt_template_path LLM/prompt_com.csv \
--mode LLMs_comm_mistral-7b \
--executable_file ../executable/linux_exec.v2.3.0.x86_64 \
--base-port 6314 \
--lm_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
--source huggingface \
--t 0.7 \
--max_tokens 256 \
--num_runs 1 \
--num-per-task 2 \
--cot \
--debug
