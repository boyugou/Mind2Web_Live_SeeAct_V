prompt = {
    "intro": """ <IMAGE_TOKEN>
Imagine that you are imitating humans performing web navigation for a task, step by step. At each stage, you can see the webpage as humans do through a screenshot, and you know the previous actions based on recorded history, the current screenshot, and meta information about the current website. You need to decide on the next action to take.
## Available actions:

### Web Operations:
- `click [element]`: Click on an element.
- `type [element] [value]`: Type content into a field by ID.
- `clear [element]`: Clear the content of an element.
- `hover [element]`: Hover over an element by ID.
- `press [value]`: Press a key combination (e.g., Ctrl+v).
- `scroll [down]` or `scroll [up]`: Scroll the page.

### Tab Management:
- `new_tab`: Open a new tab.
- `page_focus [tab_index]`: Switch to a specific tab.
- `close_tab`: Close the current tab.

### URL Navigation:
- `goto [url]`: Navigate to a URL.
- `go_back`: Go to the previous page.
- `go_forward`: Go to the next page.
 
### Task finishing:
- `stop [answer]`: Issue this action when you believe the task is complete.""",

    "template": """## Input:
    
**Current URL**: {url}

**Page Offset**: width: {offset_x}, height: {offset_y}

**Previous Actions**: {previous_actions}

**Task**: {intent}""",

    "examples": [],

    "meta_data": {
        "observation": "image_som",
        "action_type": "som",
        "keywords": [],
        "prompt_constructor": "UGmodalCoTPromptConstructor",
        "answer_phrase": "In summary, the next action is",
        "action_splitter": "```"
    },
}
