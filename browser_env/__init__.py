import asyncio
import os
from .actions import (
    Action,
    ActionParsingError,
    ActionTypes,
    action2create_function,
    action2str,
    create_check_action,
    create_click_action,
    create_focus_and_click_action,
    create_focus_and_type_action,
    create_go_back_action,
    create_go_forward_action,
    create_goto_url_action,
    create_hover_action,
    create_id_based_action,
    create_key_press_action,
    create_keyboard_type_action,
    create_mouse_click_action,
    create_mouse_hover_action,
    create_new_tab_action,
    create_none_action,
    create_page_close_action,
    create_page_focus_action,
    create_playwright_action,
    create_random_action,
    create_scroll_action,
    create_select_option_action,
    create_stop_action,
    create_type_action,
    is_equivalent,
)
from .async_envs import AsyncScriptBrowserEnv
from .envs import ScriptBrowserEnv
#from .processors import ObservationMetadata
if os.getenv('PROCESSOR_SOURCE') == 'my_processors':
    from .my_processors import ObservationMetadata
    print("using my processors")
elif os.getenv("PROCESSOR_SOURCE") == "openweb_processors":
    from .openweb_processors import ObservationMetadata
else:
    print("using original processor")
    from .original_processor import ObservationMetadata

    print("using original processors")
from .trajectory import Trajectory
from .utils import DetachedPage, StateInfo

__all__ = [
    "ScriptBrowserEnv",
    "AsyncScriptBrowserEnv",
    "DetachedPage",
    "StateInfo",
    "ObservationMetadata",
    "Action",
    "ActionTypes",
    "action2str",
    "create_random_action",
    "create_focus_and_click_action",
    "create_focus_and_type_action",
    "is_equivalent",
    "create_mouse_click_action",
    "create_mouse_hover_action",
    "create_none_action",
    "create_keyboard_type_action",
    "create_page_focus_action",
    "create_new_tab_action",
    "create_go_back_action",
    "create_go_forward_action",
    "create_goto_url_action",
    "create_page_close_action",
    "action2create_function",
    "create_playwright_action",
    "create_id_based_action",
    "create_scroll_action",
    "create_key_press_action",
    "create_check_action",
    "create_click_action",
    "create_type_action",
    "create_hover_action",
    "create_select_option_action",
    "create_stop_action",
    "ActionParsingError",
    "Trajectory",
]
