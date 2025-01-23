import os
import pdb
import re
from copy import deepcopy

Task_dict = {
    "classifieds": set(
        "87 161 134 166 135 183 151 116 189 178 185 92 163 123 64 186 9 48 111 152 127 105 202 114 94 157 198 104 162 "
        "149 182 144 57 201 74 101 146 195 30 93 72 199 5 156 206 188 207 98 187 204 125 122 82 66 68 37 173 42 109 "
        "196".split()),
    "shopping": set(
        "451 232 443 7 129 211 229 31 77 379 107 305 461 192 201 349 61 13 392 103 288 351 329 179 206 409 137 330 "
        "246 198 406 15 168 51 353 159 360 122 431 56 186 355 45 76 433 102 388 89 23 426 188 416 171 321 276 376 319 "
        "140 1 464 70 50 424 75 389 228 207 454 390 73 145 99 90 315 375 79 44 423 66 266 22".split()),
    "reddit": set(
        "19 3 142 162 116 117 28 8 176 98 202 187 73 88 100 168 194 153 144 181 2 44 7 145 118 138 4 5 31 13 169 170 "
        "52 103 128 12 26 85 102 94 146 71 203 136 195 54 192 133 173 6 32 199 167 113 204 180 84 112 183 201".split()),
    'webcanvas': set("1 3 5 7 8 9 11 12 14 19 20 21 24 26 27 28 31 32 33 34 36 37 40 41 42 44 50 51 52 53 54 55 56 57 58 59 60 62 63 66 67 70 72 73 74 75 76 77 78 79 80 81 82 84 85 86 87 88 89 90 91 93 94 95 96 98 102 103".split()),
    "webcanvas_small": set(
        "1 3 4 6 7 8 11 16 17 20 94 95 102 103 67 69 71 72 73 76 93 26 27 28 30 38 40 42 44 96 97".split()
    )
    # good cases

}
webcanvas_90 = ""
for i in range(1, 91):
    webcanvas_90 += str(i) + " "
Task_dict["webcanvs_90"] = set(webcanvas_90.split())


def count_results(file_path, task_ids):
    fail_count = 0
    pass_count = 0
    error_count = 0
    total_count = 0
    error_numbers = []
    total_steps = 0
    passed_steps = 0
    left_ids = deepcopy(task_ids)
    error_pattern = re.compile(r'\[NAME\] .*?/(\d+)\.json')
    with open(os.path.join(file_path, 'result_0.txt'), 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = error_pattern.search(line)
            if match:

                file_id = match.group(1)
                if file_id not in task_ids:
                    continue  # 跳过不在指定ID集合中的行
                left_ids.remove(file_id)
                if '[Result] (FAIL' in line:
                    fail_count += 1
                    p1 = '[Result] (FAIL with step: '
                    if p1 in line:
                        step_start = line[line.find(p1) + len(p1):]
                        p2 = ' score: '
                        step_score = step_start[:step_start.find(p2)]
                        p_s, a_s = step_score.split('/')
                        passed_steps += int(p_s.strip())
                        total_steps += int(a_s.strip())
                elif '[Result] (HALF' in line:
                    fail_count += 1
                    p1 = '[Result] (HALF with step: '
                    if p1 in line:
                        step_start = line[line.find(p1) + len(p1):]
                        p2 = ' score: '
                        step_score = step_start[:step_start.find(p2)]
                        p_s, a_s = step_score.split('/')
                        try:
                            passed_steps += int(p_s.strip())
                        except:
                            passed_steps += float(p_s.strip())
                        total_steps += int(a_s.strip())
                elif '[Result] (PASS' in line:
                    pass_count += 1
                    p1 = '[Result] (PASS with step: '
                    if p1 in line:
                        step_start = line[line.find(p1) + len(p1):]
                        p2 = ' score: '
                        step_score = step_start[:step_start.find(p2)]
                        p_s, a_s = step_score.split('/')
                        passed_steps += int(p_s.strip())
                        total_steps += int(a_s.strip())
                elif '[ERROR]' in line:
                    error_count += 1
                    error_numbers.append(file_id)

                total_count += 1

    if not left_ids:
        left_ids = []
    return fail_count, pass_count, error_count, total_count, error_numbers, left_ids, passed_steps, total_steps


# 删除指定目录中的 render_number.html 文件
def delete_render_files(directory_path, error_numbers, DELETE_ERROR_FILES):
    if DELETE_ERROR_FILES:
        for number in error_numbers:
            file_name = f'render_{number}.html'
            file_path = os.path.join(directory_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_name}")
            else:
                print(f"File {file_name} does not exist")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--task', type=str,
                        default="webcanvas", help='The task to be executed')
    parser.add_argument('--directory_path', type=str,
                        default="/root/search_agent/result/aws_perception_ngui_1epoch/baselineug")

    args = parser.parse_args()
    print(args)
    task = args.task
    DELETE_ERROR_FILES = True

    # 选择任务的ID集合
    task_ids = Task_dict[task]  # 可以替换为 shopping_ids 或 reddit_ids
    print(task_ids)
    # 统计结果
    fail_count, pass_count, error_count, total_count, error_numbers, left_ids, passed_steps, total_steps = count_results(
        args.directory_path, task_ids)
    left_ids = " ".join(left_ids)
    # 打印统计结果
    print("*" * 20)
    print(f"MODEL: {args.directory_path}")
    print(f"TASK: {task}")
    print(f"FAIL count: {fail_count}")
    print(f"PASS count: {pass_count}")
    print(f"ERROR count: {error_count}")
    print(f"Total count: {total_count - error_count}")
    print(f"Error numbers: {error_numbers}")
    print(f"Expected count: {len(task_ids)}")
    print(f"Left IDs: {left_ids}")
    print(f"TRAJ ACCURACY WITHOUT ERROR COUNT: {pass_count / (total_count - error_count) * 100:.2f}%")
    print(f"TRAJ ACCURACY WITH ERROR COUNT: {pass_count / total_count * 100:.2f}%")
    print(f"STEP SCORE WITHOUT ERROR COUNT: {passed_steps / total_steps * 100:.2f}%")

    # 删除文件
    delete_render_files(args.directory_path, error_numbers, DELETE_ERROR_FILES)
