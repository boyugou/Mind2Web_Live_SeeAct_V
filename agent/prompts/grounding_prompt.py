GROUNDING_PROMPT = """You are given a referring expression and a text observation extracted from a webpage. This observation includes interactive elements, each with an ID (e.g., [1] [A] [About]), and non-interactive elements labeled as StaticText without IDs. Your task is to identify which interactive element the referring expression points to and return the ID in the format: "In summary, the referred element is [id]."

You will also receive a screenshot where interactive elements are labeled with corresponding IDs. Use this information to determine the correct element.

An example:
- **Referring Expression**: "The sign in button on the top right of the page."
- **Text Observation**: [1] [A] [Go to Google Home.]
[2] [TEXTAREA] [Pop Workout mixSearch.q.Pop Workout mix.]
[3] [DIV] [Clear.]
[4] [DIV] [Search by voice.]
[5] [DIV] [Search by image.]
[6] [BUTTON] [Search.]
[7] [DIV] [Quick SettingsSettings.]
[] [StaticText] [Quick Settings]
[8] [A] [Google apps.]
[9] [A] [Sign inSign in.]
[] [StaticText] [Sign in]
[] [StaticText] [Filters and Topics]
[10] [A] [All]

- **Output**: "The referred element is [9]."

##Important Notes:
1. If there is no matching element, return "The referred element is not found."
2. If there is a matching element, you must return the ID (number) of the element. Do not return the text content. For example, if the referring expression is "The sign in button," you should return the ID of the button element (e.g. [9]), not the text "Sign in."

**Input:**
- **Referring Expression**: {referring_expression}

- **Text Observation**: {text_obs}

**Output Format**: "The referred element is [id]./The referred element is not found."
"""

GROUNDING_SYSTEM_PROMPT = """You are given a referring expression, a list of interactive elements with IDs, and a screenshot with labeled elements. Please return the ID of the element that the referring expression refers to."""