PORT=$1
task_ids_array=$2
aim=$3
model=$4

echo "PORT: $PORT"
echo "task_ids_array: $task_ids_array"
echo "aim: $aim"
echo "model: $model"

#PORT=23332
#task_ids_array=4
#aim=baseline
#model=aws_combined_vwa_mind2web_mathv93k_1epochs

IFS=' ' read -r -a task_ids <<< "$task_ids_array"
# shellcheck disable=SC2068
echo task_ids: ${task_ids[@]}

export DATASET="webcanvas"
# set env
export LOCAL_API_SERVER=http://127.0.0.1:12344/v1
export OPENAI_API_BASE="http://172.16.78.10:${PORT}/v1"
export PROCESSOR_SOURCE="my_processors"
export LOCAL_UG_SERVER=http://127.0.0.1:12345/v1
export PREV_ACTION_VERSION=v3
export PORT=$PORT
export OPENAI_API_KEY=""


max_steps=15
agent="prompt"  # change this to "prompt" to run the baseline without search
caption_model="none"
result_dir="result/${model}/${aim}"
instruction_path="agent/prompts/jsons/mind2web_neat.json"
# Define the batch size variable (how many examples to run before resetting the environment)
batch_size=100
python ./scripts/read_results_webcanvas.py --directory_path "$result_dir"
for ((i=0; i<${#task_ids[@]}; i+=batch_size))
do
    echo "new start"
    python ./scripts/read_results_webcanvas.py --directory_path "$result_dir"
    # Get the current batch of task_ids
    batch=("${task_ids[@]:i:batch_size}")
    # Run the python script with the current batch of task_ids
    python run_multi_single_process.py \
        --instruction_path $instruction_path \
        --model "$model" \
        --agent_type $agent \
        --result_dir $result_dir \
        --test_config_base_dir="./config_files/mind2web-live-github/test" \
        --repeating_action_failure_th 5 \
        --viewport_height 1344 \
        --viewport_width 1280 \
        --max_obs_length 8190 \
        --action_set_tag ug \
        --observation_type image_som \
        --top_p 1.0 \
        --temperature 0.1 \
        --max_tokens 4500 \
        --max_steps $max_steps \
        --value_function gpt-4o \
        --captioning_model $caption_model \
        --eval_captioning_model $caption_model \
        --task_ids "${batch[@]}" \
        --num_processes 1

    # Ensure the end index does not exceed max_idx in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
pkill
done
