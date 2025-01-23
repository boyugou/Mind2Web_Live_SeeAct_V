import argparse
import os
import pdb
import re
import json
from typing import Any, Optional
import tiktoken
from beartype import beartype
from PIL import Image
import math
from openai import OpenAI
from agent.prompts.grounding_prompt import GROUNDING_PROMPT, GROUNDING_SYSTEM_PROMPT
from agent.prompts.grounding_prompt_constructor import ground_prompt_constructor
from evaluation_harness.openai import GPTGenerator
from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
    create_coordinated_action
)
from browser_env.utils import Observation, StateInfo, pil_to_b64
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.providers.openai_utils import chat_with_ug, chat_with_141
from llms.tokenizers import Tokenizer
from browser_env.utils import *


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


gpt = GPTGenerator()


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
            self,
            test_config_file: str,
    ) -> None:
        raise NotImplementedError


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
            self,
            action_set_tag: str,
            lm_config: lm_config.LMConfig,
            prompt_constructor: PromptConstructor,
            captioning_fn=None
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn
        self.output_dict_list = []
        self.last_offset_y = 0
        # Check if the model is multimodal.
        self.multimodal_inputs = True
        self.memory = []
        self.warning = ""
        self.grounding_client = OpenAI(
            api_key="YOUR_API_KEY", base_url="http://127.0.0.1:8001/v1"
        )

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def get_images(self, trajectory, num_images):
        images = []
        # 获取指定数量的间隔图片
        for i in range(0, len(trajectory), 2):
            images.append(trajectory[i]["observation"]["raw_screenshot"])

        images = images[-num_images:]
        return images

    def call_uground(self, expression, image):

        def pil_to_b64_forug(img: Image.Image) -> str:
            with BytesIO() as image_buffer:
                img.save(image_buffer, format="PNG")
                byte_data = image_buffer.getvalue()
                img_b64 = base64.b64encode(byte_data).decode("utf-8")
            return img_b64



        response = chat_with_ug(os.environ["LOCAL_UG_SERVER"],
                                image=pil_to_b64_forug(image),
                                prompt=expression)

        print("CALLING GROUND FINISHED. Response:", response, flush=True)

        def extract_coordinates(s):
            # 使用正则表达式匹配数字
            numbers = re.findall(r'\d+', s)

            # 将字符串转换为整数
            coordinates = list(map(int, numbers))

            # 确保只返回前四个值
            return coordinates[:4]

        coordinates = response['fix_c']
        return coordinates

    def call_uground2(self, expression, image):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
  Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.

  - Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.
  - If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
  - Your answer should be a single string (x, y) corresponding to the point of the interest.

  Description: {expression}

  Answer:"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_to_b64(image)
                        }
                    }
                ]
            }
        ]

        response = chat_with_141(os.environ["LOCAL_UG_SERVER"],
                                 model="UGround",
                                 messages=messages,
                                 temperature=0.0,
                                 port=23333)

        print("CALLING GROUND FINISHED. Response:", response, flush=True)
        coordinate = eval(response["answer"])
        coordinate = (coordinate[0] / 1000, coordinate[1] / 1000)

        return coordinate

    def calculate_tokens(self, image_path, max_tokens=450, pixels_per_token=768):
        # 打开图像
        if not isinstance(image_path, Image.Image):
            image = Image.open(image_path)
        else:
            image = image_path
        width, height = image.size
        # 计算图像的总像素数
        total_pixels = width * height

        # 计算消耗的token数量
        tokens = math.ceil(total_pixels / pixels_per_token)

        if tokens > max_tokens:
            # 需要resize图像
            scale_factor = math.sqrt(max_tokens * pixels_per_token / total_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # 保持宽高比进行resize
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            (nw, nh) = resized_image.size
            assert nw * nh >= 2
            return resized_image
        else:
            # 不需要resize
            return image

    def extract_ug_values(self, action_string):

        element_description_pattern = r'"Element Description": (.*?)\n'
        action_pattern = r'"Action": (.*?)\n'
        value_pattern = r'"Value": (.*?)\n'
        element_match = re.search(element_description_pattern, action_string, re.DOTALL)
        action_match = re.search(action_pattern, action_string, re.DOTALL)
        value_match = re.search(value_pattern, action_string, re.DOTALL)
        element_description = element_match.group(1) if element_match else ""
        action = action_match.group(1) if action_match else ""
        value = value_match.group(1) if value_match else ""

        def process_output(output):
            if output.startswith("\""):
                output = output[1:]
            if output.endswith(","):
                output = output[:-1]
            if output.endswith("\""):
                output = output[:-1]
            return output

        element_description = process_output(element_description)
        action = process_output(action)
        value = process_output(value)

        try:
            if action == "":
                element_description_pattern = r'\*\*Element Description\*\*: (.*?)\n'
                action_pattern = r'\*\*Action\*\*: (.*?)\n'
                value_pattern = r'\*\*Value\*\*: (.*?)\n'
                element_match = re.search(element_description_pattern, action_string, re.DOTALL)
                action_match = re.search(action_pattern, action_string, re.DOTALL)
                value_match = re.search(value_pattern, action_string, re.DOTALL)
                element_description = element_match.group(1) if element_match else ""
                action = action_match.group(1) if action_match else ""
                value = value_match.group(1) if value_match else ""
        except:
            return element_description, action, value

        return element_description, action, value

    def find_nearest_point(self, bbox, x, y):
        def is_within_bounds(element, x, y):
            return element['left'] <= x <= element['right'] and element['top'] <= y <= element['bottom']

        def calculate_distance(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        nearest_point = None
        min_distance = float('inf')
        Interactable_bbox = []
        bbox = bbox.to_dict(orient='records')
        for b in bbox:
            interactable = b["Interactable"]
            top, right, bottom, left = b["Top"], b["Right"], b["Bottom"], b["Left"]
            if interactable:
                Interactable_bbox.append({"top": top, "right": right, "bottom": bottom, "left": left})
        if len(Interactable_bbox) == 0:
            print("length of Interactable_bbox is 0")
            return (x, y)
        for element in Interactable_bbox:
            if is_within_bounds(element, x, y):
                print("WITHIN BOUNDS")
                center_x = (element['left'] + element['right']) / 2
                center_y = (element['top'] + element['bottom']) / 2
                return (center_x, center_y)

            # Calculate the center point of the rectangle
            center_x = (element['left'] + element['right']) / 2
            center_y = (element['top'] + element['bottom']) / 2

            # Calculate distance from the point to the center of the rectangle
            distance = calculate_distance(x, y, center_x, center_y)

            if distance < min_distance:
                min_distance = distance
                nearest_point = (center_x, center_y)
        print("NEAREST_POINT", nearest_point)
        return nearest_point

    def call_atlas(self, expression, image):

        def calculate_midpoint(x0, y0, x1, y1):
            midpoint_x = (x0 + x1) / 2
            midpoint_y = (y0 + y1) / 2
            midpoint_x = midpoint_x / 1000
            midpoint_y = midpoint_y / 1000
            return (midpoint_x, midpoint_y)

        def pil_to_b64_for_atlas(img: Image.Image) -> str:
            with BytesIO() as image_buffer:
                img.save(image_buffer, format="PNG")
                byte_data = image_buffer.getvalue()
                img_b64 = base64.b64encode(byte_data).decode("utf-8")
            return img_b64

        content = [
            {
                "type": "text",
                'text': f'<IMAGE_TOKEN>\nIn the screenshot of this web page, please give me the coordinates of the element (with point).\n{expression}.',
            },

        ]
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64_for_atlas(image),
                        "max_dynamic_patch": 16

                    },
                }
            ]
        )
        user_input = {
            "role": "user",
            "content": content
        }
        messages = [user_input]

        response = chat_with_141(os.environ["LOCAL_UG_SERVER"],
                                 model="/disk2/OS-Atlas-Base-7B/Qwen2-VL-7B-Instruct",
                                 messages=messages,
                                 temperature=0.01,
                                 max_tokens=512,
                                 port=8000)
        response = response["answer"]
        coordinates = re.findall(r'\((\d+),(\d+)\)', response)
        coordinates = [(int(x), int(y)) for x, y in coordinates]
        x0, y0, x1, y1 = coordinates[0][0], coordinates[0][1], coordinates[1][0], coordinates[1][1]
        coordinates = calculate_midpoint(x0, y0, x1, y1)
        return coordinates

    @beartype
    def next_action(
            self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any],
            images=None,
            output_response: bool = False, **kwargs
    ) -> Action:
        # Create page screenshot image for multimodal models.
        # images = [self.calculate_tokens(image, 600) for image in images]
        GROUNDING = "UG"

        page_screenshot_list = self.get_images(trajectory, 1)
        page_screenshot_list = [Image.fromarray(image) for image in page_screenshot_list]
        bboxes = trajectory[-1]["observation"]["bboxes"]
        prompt = self.prompt_constructor.construct(
            trajectory, intent, page_screenshot_list,
            images, meta_data, warnings=self.warning,
            **kwargs
        )

        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt, num_outputs=1)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                print(f'Agent: {response}', flush=True)
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                parsed_response = parsed_response.strip()
                expression, action, value = self.extract_ug_values(parsed_response)
                expression = expression.strip()
                action = action.strip()
                value = value.strip()
                action_coordinate = None

                print(expression, action, value, flush=True)
                raw_action = action
                if action in ["memorize"]:
                    self.memory.append(value)
                    while action != "memorize":
                        prompt = self.prompt_constructor.construct(
                            trajectory, intent, page_screenshot_list,
                            images, meta_data, memory=self.memory, **kwargs
                        )
                        response = call_llm(lm_config, prompt, num_outputs=1)
                        force_prefix = self.prompt_constructor.instruction[
                            "meta_data"
                        ].get("force_prefix", "")
                        response = f"{force_prefix}{response}"
                        if output_response:
                            print(f'Agent: {response}', flush=True)
                        parsed_response = parsed_response.strip()
                        expression, action, value = self.extract_ug_values(parsed_response)
                        expression = expression.strip()
                        action = action.strip()
                        value = value.strip()
                action = action.lower()

                #     action_set = ["click", "clear", "hover", "type", "scroll", "scroll [down]", "scroll [up]",
                #                  "press", "new_tab", "page_focus", "close_tab", "goto"]

                if action == "scroll down":
                    action = "scroll [down]"
                if action == "scroll up":
                    action = "scroll [up]"

                if action == "scroll" and "up" in value:
                    action = "scroll [up]"
                if action == "scroll" and "down" in value:
                    action = "scroll [down]"

                if action in ["click", "clear", "hover", "type"]:

                    # som_image = trajectory[-1]["observation"]["image"]
                    # id2center = trajectory[-1]["observation"]["id2center"]
                    # text_obs = trajectory[-1]["observation"]["text"]
                    # grounding_prompt = GROUNDING_PROMPT.format(referring_expression=expression,
                    #                                            text_obs=text_obs)
                    # grounding_prompt = ground_prompt_constructor(grounding_prompt, Image.fromarray(som_image), GROUNDING_SYSTEM_PROMPT)
                    # grounding_response = gpt.generate(grounding_prompt)
                    #
                    # matches = re.findall(r'\[(.*?)\]', grounding_response)
                    # print(f"GROUNDING RESPONSE: {grounding_response}, match: {matches}", flush=True)
                    # try:
                    #     action_coordinate = (id2center[matches[0]][0] / trajectory[-1]['info']['viewport_info']['width'],
                    #                          id2center[matches[0]][1] / trajectory[-1]['info']['viewport_info']['height'])
                    #
                    # except:
                    #     print("using uground")
                    #     x0, y0, x1, y1 = self.call_uground(expression, page_screenshot_list[-1])
                    if GROUNDING == "UG":
                        action_coordinate = self.call_uground(expression, page_screenshot_list[-1])
                        action_coordinate = (action_coordinate[0] / page_screenshot_list[-1].size[0],
                                             action_coordinate[1] / page_screenshot_list[-1].size[1])
                        print("UGROUND_COORD", action_coordinate)

                    elif GROUNDING == "ATLAS":
                        action_coordinate = self.call_atlas(expression, page_screenshot_list[-1])

                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "ug":
                    action = create_coordinated_action(action, value, action_coordinate)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response

                # output_dict = process_rationale(response)
                self.output_dict_list.append((response, f"{raw_action} [{expression}] [{value}]"))
                break
            except ActionParsingError as e:
                print("ACTION PARSING ERROR", e, flush=True)
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    self.output_dict_list.append("None")
                    break
        return action

    def reset(self, test_config_file: str) -> None:
        self.output_dict_list = []
        self.memory = []
        self.warning = ""


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        pass
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        # tokenizer = Tokenizer(args.provider, args.model)
        tokenizer = None
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )

    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent


def get_input():
    lines = []
    print('Please Input Action', flush=True)
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return '\n'.join(lines)
