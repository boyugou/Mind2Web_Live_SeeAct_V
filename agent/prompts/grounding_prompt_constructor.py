from browser_env.utils import pil_to_b64
from PIL import Image


def resize_image_with_max_dimension(image, max_dimension):
    # 获取原始尺寸
    original_width, original_height = image.size

    # 确定缩放比例
    if original_width > max_dimension or original_height > max_dimension:
        scaling_factor = min(max_dimension / original_width, max_dimension / original_height)
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # 缩放图像
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def ground_prompt_constructor(prompt, screenshot, system_message=None):
    content = [
        {
            "type": "text",
            "text": prompt
        },

    ]
    screenshot = resize_image_with_max_dimension(screenshot, 1024)

    content.extend(
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "max_dynamic_patch": 18

                },
            }
        ]
    )

    user_input = {
        "role": "user",
        "content": content
    }
    system_message = {
        "role": "system",
        "content": system_message
    }
    message = [system_message, user_input]

    return message
