import pdb
import json
from typing import List, Any
import os
import tempfile
import subprocess
import time
from gymnasium import Env
from browser_env.actions import Action, execute_action, get_action_space
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env import (
    ScriptBrowserEnv,

)
from browser_env.processors import ObservationHandler
from pathlib import Path
from browser_env.utils import Observation
from convert_traj import add_selected_tag, get_element_id
from agent.prompts.prompt_constructor import MultimodalCoTPromptConstructor
from browser_env.actions import create_id_based_action
import sys
sys.path.append('.')

prompt_constructor = MultimodalCoTPromptConstructor(
            instruction_path=r'/data/users/zhangjunlei/code/search_agent/agent/prompts/jsons/p_som_cot_id_actree_3s.json',
            lm_config=None,
            tokenizer=None,)



class TransferEnv(ScriptBrowserEnv):
    def __init__(self, **kwargs):
        super(TransferEnv, self).__init__(**kwargs)

    def reset_observation_type(self, observation_type, captioning_fn):
        match observation_type:
            case "html" | "accessibility_tree" | "accessibility_tree_with_captioner":
                self.text_observation_type = observation_type
                self.image_observation_type = ""
                self.main_observation_type = "text"
            case "image":
                self.image_observation_type = observation_type
                self.text_observation_type = ""  # type: ignore[assignment]
                self.main_observation_type = "image"
            case "image_som":
                self.image_observation_type = observation_type
                self.text_observation_type = observation_type  # type: ignore[assignment]
                self.main_observation_type = "image"
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )
        self.observation_handler = ObservationHandler(
            self.main_observation_type,
            self.text_observation_type,
            self.image_observation_type,
            self.current_viewport_only,
            self.viewport_size,
            captioning_fn,
        )

        self.observation_space = (
            self.observation_handler.get_observation_space()
        )



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
def set_web_state(previous_actions,
                  config_file,
                  viewport_size,
                  caption_image_fn):
    """
    Sets the state of a web page using Playwright.

    Parameters:
    - page (object): The Playwright page object.
    - previous_actions (list[str]): A list of strings representing previous actions performed.
    - config_file (str): Path to the test configuration file. Located within `config_files/vwa/{task}/**.json`.
    """
    with open(config_file, 'r') as f:
        _c = json.load(f)
        intent = _c["intent"]
        task_id = _c["task_id"]
        image_paths = _c.get("image", None)
        images = []

        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()
            # subprocess to renew the cookie
            subprocess.run(
                [
                    "python",
                    "browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ]
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(config_file, "w") as f:
                json.dump(_c, f)
    if image_paths is not None:
        intent_img_des = caption_image_fn([image_paths])
    else:
        intent_img_des = None


    env = TransferEnv(
        headless=True,
        slow_mo=0,
        observation_type="image_som",
        current_viewport_only=True,
        viewport_size={
            "width": viewport_size[0],
            "height": viewport_size[1],
        },
        save_trace_enabled=False,
        sleep_after_execution=0.,
        captioning_fn=caption_image_fn
    )

    # start_time = time.time()
    obs, info = env.reset(options={"config_file": config_file})
# 'In summary, the next action I will perform is```type [6] [Cell phones] ```'
    #previous_actions = ["type [6] [Cell phones]", "click [7]"]
    for a_hist in previous_actions:
        a_hist = a_hist["action"]
        response = f"In summary, the next action I will perform is```{a_hist}```"
        parsed_response = prompt_constructor.extract_action(
            response
        )
        action = create_id_based_action(parsed_response)
        page = env.step(action)

    return env, page
def save_page_content(page, file_name: str = 'page_content.html'):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(page.content())
        print(f'页面内容已保存到: {file_name}')
def main(data_dir):
    data = []
    with open(data_dir, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    for d in data:
        if "shopping" not in d['file_name']:
            continue
        intent_image = d['intent_image']["url"]
        previous_actions = d['previous_actions']
        if not previous_actions:
            continue

        file_name = d['file_name']
        file_type = file_name.split("_")
        file_id = file_type[-1]
        file_type = file_type[:-1]
        file_type = "_".join(file_type)
        config_path = f'config_files/vwa/{file_type}/{file_id}.json'
        env, page = set_web_state(previous_actions, config_path, d['viewport'], caption_request)
        use_id, element_id = get_element_id(action=d['action'], page=page, data=d)
        add_selected_tag(page=page, element_id=int(element_id))
        env.reset_observation_type("accessibility_tree_with_captioner", caption_request)

        obs = env._get_obs()

        a = 1





if __name__ == "__main__":
    data_dir = r"/data/users/zhangjunlei/download/dataset/traj_data/new_data_json/data_merge_724.json"
    main(data_dir)

