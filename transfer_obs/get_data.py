import pickle
import re
import pandas as pd
import requests
import json
import pdb
from typing import List

def open_pickle(file_name):
    with open(file_name, "rb") as f:
        data_list = pickle.load(f)
    return data_list


def eval_caption_image_fn(images, prompt):
    url = "http://127.0.0.1:5001/process"
    if prompt is None:
        # Perform VQA
        prompt = [
            "Please describe the image." for _ in range(len(images))
        ]
        data = {
            "image_urls": images,
            "questions": prompt,
            "max_new_tokens": 100
        }

    else:
        data = {
            "image_urls": images,
            "questions": prompt,
            "max_new_tokens": 100
        }
    response = requests.post(url, json=data)
    # print(response.json())
    ans = []
    for a in response.json():
        ans.append(a["answer"])
    return ans


import re


def replace_image_url_from_text(text):
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(pattern, text)
    # print(urls)
    image_des = eval_caption_image_fn([urls[0]], ['Please describe the image.'])
    text = text.replace(urls[0], '')
    text = text.replace('IMG_URL_ANNO', image_des[0])
    return text


def textobs_df(text_obs, add_coordinate):
    pattern = re.compile(r'(\[-?\d*\])? \[(\w+)\] \[\[(-?\d+), (-?\d+), (-?\d+), (-?\d+)\] ([^\]]+)\]')
    pattern1 = re.compile(r'(\[-?\d*\])? \[(\w+)\] \[\[(-?\d+), (-?\d+), (-?\d+), (-?\d+)\] \]')
    parsed_data = []
    new_text_obs = ''
    for line in text_obs.strip().split('\n'):

        line = line.replace('[IMG_URL_ANNO]', '(IMG_URL_ANNO)')
        # print(f"origin:{line}")
        match = pattern.match(line)
        match1 = pattern1.match(line)
        if match:
            index, element_type, x, y, x_end, y_end, text = match.groups()
            # print(f"new: {match}")
        elif match1:
            index, element_type, x, y, x_end, y_end = match1.groups()
            text = ''
            # print(f"new: {match1}")
        else:
            text = line
            print(f"match error: {line}")
        text = text.replace('\n', ' ')
        # if element_type == 'IMG':
        #     # print('yes')
        #     try:
        #         text = replace_image_url_from_text(text)
        #     except:
        #         print('image url error')
        #         text = text
        if match or match1:
            x_cen = (int(x) + int(x_end)) / 2
            y_cen = (int(y) + int(y_end)) / 2
            width_X = int(x_end) - int(x)
            height_y = int(y_end) - int(y)
            if add_coordinate:
                new_text_obs_line = f"{index} [{{'center':[{x_cen},{y_cen}], 'size':[{width_X},{height_y}]}}] [{element_type}] [{text}]"
            else:
                new_text_obs_line = f"{index} [{element_type}] [{text}]"
            new_text_obs += new_text_obs_line
            new_text_obs += '\n'

        # if index != '[]':
            parsed_data.append({
                "Index": index if index else "N/A",
                "Element Type": element_type,
                "center": f"{x_cen},{y_cen}",
                "size": f"{width_X},{height_y}",
                "Text": text
            })
        else:
            parsed_data.append({
                "Index": "N/A",
                "Element Type": "[STATIC_TEXT]",
                "center": "",
                "size": "",
                "Text": text
            })
        # print('\n')

    # 转换为DataFrame并显示
    df = pd.DataFrame(parsed_data)
    return df, new_text_obs


def get_action_text(textobs_df, action):
    # nums = get_nums(action)
    pattern = re.compile(r'\[(\d+)\]')

    # 使用findall方法查找所有匹配项
    nums = pattern.findall(action)

    # print(nums)
    if len(nums) == 0:
        return '', '-1,-1', '-1,-1'
    match_str = '[' + nums[0] + ']'
    for index, row in textobs_df.iterrows():
        if row['Index'] == match_str:
            return row['Text'], row['center'], row['size']

    print('no match from get action text')
    return '', '-1,-1', '-1,-1'


'''
def gen_one_file(file_name, write_path):
    data_list = open_pickle(file_name)
    history_actions = []
    for item in data_list:
        text_df, clear_text_obs = textobs_df(item['text_obs'])
        action_text, action_center, action_size = get_action_text(text_df, item['action'])
        # print(action_text, action_center, action_size)
        x_cen, y_cen = [x  for x in action_center.split(',')]
        w, h = [x for x in action_size.split(',')]

        json_dict = {}
        json_dict['text_obs'] = clear_text_obs
        json_dict['action'] = item['action'].split('where')[0].replace('\n', '')
        json_dict['intent'] = item['task']
        json_dict['URL'] = item['url']
        json_dict['viewport'] = [1344, 1280]
        json_dict['offset'] = [item['offset_height'], item['offset_width']]
        json_dict['action_center'] = [x_cen, y_cen, h, w]
        json_dict['previous_actions'] = history_actions
        preaction_des = {"action": item['action'], "descreption": action_text, "action center": [x_cen, y_cen, h, w]}
        history_actions.append(preaction_des)

        print(json_dict['action'])
        with open(write_path, "a+") as f:
            json.dump(json_dict, f)
            f.write("\n")
    print('\n')
'''

def caption_request(
        images: List[str],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
        server="http://127.0.0.1:5001/process"
) -> List[str]:
    import requests
    if prompt is None:
        # Perform VQA
        prompt = [
            "Please describe the image in detail." for _ in range(len(images))
        ]
        data = {
            "image_urls": images,
            "questions": prompt,
            "max_new_tokens": max_new_tokens
        }

    else:
        data = {
            "image_urls": images,
            "questions": prompt
        }
    response = requests.post(server, json=data)
    captions = []
    for a in response.json():
        captions.append(a["answer"])
    return captions

# import json
def gen_one_file(file_name, write_path, subdir):
    data_list = open_pickle(file_name)
    history_actions = []
    _, sort_name, index = subdir.split('_')
    index = int(index)
    print(sort_name, index)
    sort_name_json = f"/data/users/zhangjunlei/download/dataset/traj_data/config_json/test_{sort_name}.json"
    with open(sort_name_json, 'r') as f:
        data = json.load(f)
        # print(len(data))
    # if data[index]['image']:
    #     try:
    #         intent_img_des = eval_caption_image_fn([data[index]['image']], ['Please describe the image.'])
    #     except:
    #         print('image_url_error')
    #         intent_img_des = ''
    # else:
    #     intent_img_des = ''
    # print(intent_img_des)
    # print(data[index]['image'])
    add_coordinate = True
    output_list_dict = []
    for item in data_list:
        text_df, clear_text_obs = textobs_df(item['text_obs'], add_coordinate=add_coordinate)
        action_text, action_center, action_size = get_action_text(text_df, item['action'])
        # print(action_text, action_center, action_size)
        x_cen, y_cen = [x for x in action_center.split(',')]
        w, h = [x for x in action_size.split(',')]

        # print(data[index]['image'])
        json_dict = {}
        try:
            json_dict['intent_image'] = {'url': data[index]['image'], 'description': caption_request([data[index]['image']], prompt=["Please describe the given image in detail."]) }
        except:

            json_dict['intent_image'] = {'url': "", 'description:': ""}
        json_dict['text_obs'] = clear_text_obs
        json_dict['action'] = item['action'].split('where')[0].replace('\n', '')
        json_dict['intent'] = item['task']
        json_dict['URL'] = item['url']
        json_dict['viewport'] = {"weight": 1344, "height": 1280}
        json_dict['offset'] =  {"offset_width":item['offset_width'],
                                "offset_height":item['offset_height']}

        json_dict['action_center'] = [x_cen, y_cen, h, w]
        json_dict['previous_actions'] = history_actions[:]
        json_dict["file_name"] = file_name.split('/')[-1].split('.')[0]
        preaction_des = {"action": item['action'].split('where'
                                                        )[0].replace('\n', ''), "description": action_text,
                         "action center": [x_cen, y_cen, h, w]}

        history_actions.append(preaction_des)
        json_dict["som_image"] = item['som_image']
        json_dict['image'] = item['image']
        json_dict['step'] = item['step']
        output_list_dict.append(json_dict)
        # print(json_dict['action'])
    return output_list_dict


if __name__ == "__main__":

    save_name = "data_merge_724"
    # save_name = "data_zijie_719"
    write_path = f"/data/users/zhangjunlei/download/dataset/traj_data/new_data_json//{save_name}.json"
    log_path = f"./log/{save_name}.log"
    # file_name = "traj/test_classifieds_0/test_classifieds_0.pkl"

    import os

    base_dir = os.path.join("/data/users/zhangjunlei/download/dataset/traj_data", "traj")
    # base_dir = os.path.join(".", "jyb", "traj")
    dirlist = os.listdir(base_dir)
    all_list = []
    for subdir in dirlist:

        file_name = os.path.join(base_dir, f"{subdir}", f"{subdir}.pkl")
        all_list.append(gen_one_file(file_name, write_path, subdir))

    with open("vwa_data_0729.pkl", "wb") as file:
        pickle.dump(all_list, file)
