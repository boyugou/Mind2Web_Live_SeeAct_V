import json
import re
from pathlib import Path
from typing import Any, TypedDict
from PIL import Image
import pdb
from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
import requests
import os


class Instruction(TypedDict):
    """Instruction for constructing prompt"""

    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))

        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
            self, intro: str, examples: list[tuple[str, str]], current: str
    ) -> APIInput:

        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        elif "huggingface" in self.lm_config.provider:
            # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
            if "Llama-2" in self.lm_config.model:
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                                   (
                                       B_SYS + intro + E_SYS + examples[0][0],
                                       examples[0][1],
                                   )
                               ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")
            else:
                raise ValueError(
                    f"Huggingface models do not support model_tag {self.lm_config.gen_config['model_tag']}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        if os.getenv("DATASET",'None')=='webcanvas':
            return url
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )


class MultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            page_screenshot_img: Image.Image,
            images: list[Image.Image],
            meta_data: dict[str, Any] = {},
            relection: str = "",
            **kwargs
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url

        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )
        if relection != '':
            current += f"\n\n{relection}"

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    def get_lm_api_input(
            self,
            intro: str,
            examples: list[tuple[str, str, str]],
            current: str,
            page_screenshot_img: Image.Image,
            images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        INTERNVL = False
        LLAMA3 = True
        message: list[dict[str, str]] | str | list[str | Image.Image]

        if "openai" in self.lm_config.provider:

            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                            "name": "example_user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i + 2}) input image {image_i + 1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})
                return message
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
        elif "google" in self.lm_config.provider:
            if self.lm_config.mode == "completion":
                message = [
                    intro,
                    "Here are a few examples:",
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(f"Observation\n:{x}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(example_img),
                        ]
                    )
                    message.append(f"Action: {y}")
                message.append("Now make prediction given the observation")
                message.append(f"Observation\n:{current}\n")
                message.extend(
                    [
                        "IMAGES:",
                        "(1) current page screenshot:",
                        pil_to_vertex(page_screenshot_img),
                    ]
                )
                for image_i, image in enumerate(images):
                    message.extend(
                        [
                            f"({image_i + 2}) input image {image_i + 1}",
                            pil_to_vertex(image),
                        ]
                    )
                message.append("Action:")
                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )


class InternVLMultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            page_screenshot_img: Image.Image,
            images: Any,
            meta_data: dict[str, Any] = {},
            relection: str = "",
            viewport_info: dict = {},
    ) -> APIInput:
        # ["text_obs","url", "viewport_width","viewport_height","offset_x","offset_y","intent", "previous_actions"]
        # import pdb;pdb.set_trace()
        intro = self.instruction["intro"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]

        def process_data(data):
            data = re.sub(r', url: [^\]]+', '', data)
            data = re.sub(r'description: (.+?)\]', r'(description: \1)]', data)
            return data

        obs = process_data(obs)

        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        # import pdb; pdb.set_trace()
        prev_actions = meta_data["action_history"]
        prev_actions = [p for p in prev_actions if type(p) == dict]
        prev_actions_str = ""
        for prev_action in prev_actions:
            tmp_str = ""
            tmp_str += f"action: {prev_action['action']}"
            if prev_action['description']:
                item_id = extract_number_with_brackets(prev_action['action'])
                tmp_str += f", where {item_id} is: {prev_action['description']}."
            prev_actions_str += tmp_str + "\n"

        current = template.format(
            text_obs=obs,
            url=self.map_url_to_real(url),
            viewport_width=viewport_info['width'],
            viewport_height=viewport_info['height'],
            offset_x=viewport_info['offsetX'],
            offset_y=viewport_info['offsetY'],
            intent=intent,
            previous_actions=prev_actions_str,
        )
        if relection != '':
            current += f"\n\n{relection}"

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, current, page_screenshot_img
        )

        return prompt

    def get_lm_api_input(
            self,
            intro: str,
            current: str,
            page_screenshot_img: Image.Image,
    ) -> APIInput:
        user_text = intro + '\n' + current
        user_input = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(page_screenshot_img)
                    },
                },
            ]
        }

        message = [user_input]
        return message

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            text = match.group(1)
            if text[0] == '{':
                text = text[1:]
            while text[-1] == ' ' or text[-1] == '}':
                text = text[:-1]
            return text
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


def extract_number_with_brackets(operation):
    import re
    match = re.search(r'(\[\d+\])', operation)
    if match:
        return match.group(1)  # 返回匹配到的带中括号的数字
    else:
        return None  # 如果没有匹配到带中括号的数字，返回Non


class UGmodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            page_screenshot_list: list[Image.Image],
            images: Any,
            meta_data: dict[str, Any] = {},
            viewport_info: dict = {},
            **kwargs
    ) -> APIInput:
        QWEN2 = True

        memory = []
        if "memory" in kwargs:
            memory = kwargs["memory"]

        intro = self.instruction["intro"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        if len(trajectory) > 1:
            previous_actions = trajectory[-2]["pre_actions_str"]
        else:
            previous_actions = "No previous actions."

        page = state_info["info"]["page"]
        url = page.url

        is_scroll_at_bottom = state_info["info"]["is_scroll_at_bottom"]
        is_scroll_at_top = state_info["info"]["is_scroll_at_top"]
        current = template.format(
            url=self.map_url_to_real(url),
            offset_x=viewport_info['offsetX'],
            offset_y=viewport_info['offsetY'],
            intent=intent,
            previous_actions=previous_actions
        )
        if is_scroll_at_bottom or is_scroll_at_top:
            current += "\n\n**Scroll Status**: "
            if is_scroll_at_bottom:
                current += "You are at the bottom of the page. You can not conduct `scroll [down]` action now. "
            if is_scroll_at_top:
                current += "You are at the top of the page. You can not conduct `scroll [up]` action now."

        assert all([f"{{k}}" not in current for k in keywords])
        if QWEN2:

            prompt = self.get_qwen_api_input(
                intro, current, page_screenshot_list, images, intent, memory)
        else:
            prompt = self.get_lm_api_input(
                intro, current, page_screenshot_list, images, intent, memory)
        return prompt

    def get_lm_api_input(
            self,
            intro: str,
            current: str,
            page_screenshot_list: list[Image.Image],
            images: Any,
            intent: Any='None',
            memory: list[str] = [],
            raw_intent: str = ""
    ) -> APIInput:
        user_text = intro + '\n\n' + current
        user_text = user_text.strip()
        memory_str = ""

        for i, mem in enumerate(memory):
            memory_str += f"Memory {i + 1}: {mem}\n"
        if memory_str:
            user_text += "\n"
            user_text += memory_str
        content = [
            {
                "type": "text",
                "text": user_text
            },

        ]

        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(page_screenshot_list[-1]),
                        "max_dynamic_patch": 16

                    },
                }
            ]
        )

        for i, image in enumerate(images):
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(image),
                        "max_dynamic_patch": 2
                    },
                }])

        user_input = {
            "role": "user",
            "content": content
        }

        if raw_intent or intent:
            system_message = {
                "role": "system",
                "content": f"You are a web agent, navigating through the web browser to complete a task. The task is: {raw_intent}" if raw_intent else f"The task is: {intent}."
            }
            message = [system_message, user_input]
        else:
            message = [user_input]
        return message

    def get_qwen_api_input(
            self,
            intro: str,
            current: str,
            page_screenshot_list: list[Image.Image],
    ) -> APIInput:
        user_text = intro + '\n\n' + current
        user_text = user_text.strip()
        memory_str = ""


        content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(page_screenshot_list[-1]),
                    },
                }
            ]

        content.extend([
            {
                "type": "text",
                "text": user_text
            },

        ])


        user_input = {
            "role": "user",
            "content": content
        }

        message = [user_input]
        return message


    def _extract_action(self, response: str) -> str:
        return response


class UG_QWEN_promptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
            self,
            instruction_path: str | Path,
            lm_config: lm_config.LMConfig,
            tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
            self,
            trajectory: Trajectory,
            intent: str,
            page_screenshot_list: list[Image.Image],
            images: Any,
            meta_data: dict[str, Any] = {},
            viewport_info: dict = {},
    ) -> APIInput:
        # 
        '''czj: I assert the way the following elements are acquired are correct. And use a list named "used_images" to collect the images used tas input.
                Then the "current" string is split according to the "image_placeholder" which is the same as used before "<IMAGE_TOKEN>". 
                Then multiple image inputs are appended in a list according to where the image should be placed. 
                See function "get_lm_api_input" for implementation, and "https://github.com/QwenLM/Qwen2-VL#add-ids-for-multiple-image-inputs" for reference
        elements:
            url=self.map_url_to_real(url),
            viewport_width=viewport_info['width'],
            viewport_height=viewport_info['height'],
            offset_x=viewport_info['offsetX'],
            offset_y=viewport_info['offsetY'],
            intent=intent,
            screenshot=screenshots_str,
            previous_actions=previous_action_str,
            sub_tasks=cur_plan
        '''
        used_images = []
        image_placeholder = '<IMAGE_TOKEN>'
        #
        num_steps = int((len(trajectory) - 1) / 2)

        if len(trajectory) > 1:
            previous_actions = trajectory[-2]["pre_actions_str"]
        else:
            previous_actions = "No previous actions."

        intro = self.instruction["intro"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        page = state_info["info"]["page"]
        url = page.url
        for im in range(len(images)):
            intent = intent + f" {image_placeholder} "
            used_images.append(images[im])

        screenshots_str = ""

        for i in range(len(page_screenshot_list)):
            if i == len(page_screenshot_list) - 1:
                screenshots_str += f"Screenshot of current page: {image_placeholder}\n"
            else:
                screenshots_str += f"Screenshot from {len(page_screenshot_list) - 1 - i} pages ago: {image_placeholder}\n"
            used_images.append(page_screenshot_list[i])

        current = template.format(
            url=self.map_url_to_real(url),
            viewport_width=viewport_info['width'],
            viewport_height=viewport_info['height'],
            offset_x=viewport_info['offsetX'],
            offset_y=viewport_info['offsetY'],
            intent=intent,
            screenshots=screenshots_str,
            previous_actions=previous_actions,
        )

        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(
            intro, current, used_images, image_placeholder
        )

        return prompt

    def get_lm_api_input(
            self,
            intro: str,
            current: str,
            used_images: list[Image.Image],
            image_placeholder: str
    ) -> APIInput:
        user_text = intro + '\n'

        current_splits = current.split(image_placeholder)
        current_start = current_splits[0]
        current_lists = current_splits[1:]

        user_text = user_text + current_start
        content = [
            {
                "type": "text",
                "text": user_text
            },
        ]
        for cur_str, cur_img in zip(current_lists, used_images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(cur_img)
                    }
                },
            )
            content.append(
                {
                    "type": "text",
                    "text": cur_str
                }
            )

        user_input = {
            "role": "user",
            "content": content
        }

        message = [user_input]
        return message

    def _extract_action(self, response: str) -> str:
        return response
