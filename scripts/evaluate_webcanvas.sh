#!/bin/bash
aim=baselineug
model="aws_perception_ngui_1epoch"


#webcanvas_id=("1 3 5 7 8 9 11 12 14 19 20 21 24 26 27 28 31 32 33 34 36 37 40"
#              "57 58 59 60 62 63 66 67 70 72 73 74  85 86 87 88 89 90 91 93 94 95 96"
#              "41 42 44 50 51 52 53 54 55 56 75 76 77 78 79 80 81 82 84  98 102 103")

# J-6x7,ZQs#*rcd9C
webcanvas_id=("1 3 5 7 8 9 11 12 14 19 20 21 24 26 27 28 31 32 33"
               "34 36 37 40  57 58 59 60 62 63 66 67 70 72 73 74"
              "41 42 44 50 51 52 53 54 55 56 75 76 77 78 79 80 81"
               "82 84 85 86 87 88 89 90 91 93 94 95 96 98 102 103")

#webcanvas_id=("1 3 5 7 8 9 11 12 14 19 20 21 24 26 27 28 31 32 33 34 36 37 40 41 42 44 50 51 52 53 54 55 56 57 58 59 60 62 63 66 67 70 72 73 74 75 76 77 78 79 80 81 82 84 85 86 87 88 89 90 91 93 94 95 96 98 102 103")


# CUDA_VISIABLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --served-model-name UGround --model /disk2/models/UGround-V1-7B/Qwen2-VL-7B-Instruct  --dtype bfloat16 --port 23333 --gpu_memory_utilization 0.6
PORT=23334 # aws_mathv360kbash

for i in "${!webcanvas_id[@]}"; do
   nohup bash ./scripts/test_webcanvas.sh "${PORT}" "${webcanvas_id[i]}" "${aim}"  "${model}" > "log/${aim}_${model}_webcanvas_${i}.txt" &
done

#for i in "${!webcanvas_id[@]}"; do
#    bash ./scripts/test_webcanvas.sh "${PORT}" "${webcanvas_id[i]}" "${aim}"  "${model}"
#done