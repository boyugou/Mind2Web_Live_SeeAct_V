import os
import pdb
import random


def find_missing_numbers(base_path, task_name, max_value):
    # 获取所有文件夹名称
    folders = os.listdir(base_path)

    # 找到以指定任务名开头的文件夹，并提取它们的数字部分
    existing_numbers = set()
    for folder in folders:
        if folder.startswith(task_name):
            try:
                number = int(folder.split('_')[-1])
                existing_numbers.add(number)
            except ValueError:
                pass

    # 找到从 1 到 max_value 之间不存在的数字
    missing_numbers = [i for i in range(1, max_value + 1) if i not in existing_numbers]

    # 打乱顺序
    random.shuffle(missing_numbers)

    return missing_numbers


# 示例使用
base_path = '/data/users/zhangjunlei/download/dataset/traj_data/traj/'
task_name = 'test_reddit'
max_value = 210
# reddit 210
missing_numbers = find_missing_numbers(base_path, task_name, max_value)
missing_numbers = [str(m) for m in missing_numbers]
exist_numbers = "19 3 142 162 116 117 28 8 176 98 202 187 73 88 100 168 194 153 144 181 2 44 7 145 118 138 4 5 31 13 169 170 52 103 128 12 26 85 102 94 146 71 203 136 195 54 192 133 173 6 32 199 167 113 204 180 84 112 183 201"
exist_numbers = exist_numbers.split(" ")

print(len(exist_numbers))
missing_numbers = [m for m in missing_numbers if m not in exist_numbers]

output = ' '.join(missing_numbers)
output = output.split(" ")
output = output[:(60-len(exist_numbers))]
output = ' '.join(output)
print(output)
