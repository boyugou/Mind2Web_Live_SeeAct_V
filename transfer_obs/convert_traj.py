import numpy as np
import os
from playwright.sync_api import Page
from transfer_obs.processors import get_page_bboxes, draw_bounding_boxes
import pdb
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO, StringIO
def save_page_content(page, file_name: str = 'page_content.html'):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(page.content())
        print(f'页面内容已保存到: {file_name}')



def add_selected_tag(page, element_id, tag_str="[SELECTED_ELEMENT]") -> str:
    """JavaScript code to return bounding boxes and other metadata from HTML elements."""
    js_script = f"""
           (() => {{
               const interactableSelectors = [
                   'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                   '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                   '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                   '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'
               ];

               const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
               const modifiedTextSelectors = textSelectors.map(selector =>
                   `:not(${{interactableSelectors.join(', ')}}):not(style) > ${{selector}}`
               );

               const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
               const elements = document.querySelectorAll(combinedSelectors.join(', '));

               const pixelRatio = window.devicePixelRatio;
               let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable,Title,ImageUrl,RelX0,RelY0,RelX1,RelY1\\n";

               let counter = 1;
               let imageUrl = ''
               elements.forEach(element => {{
                   const rect = element.getBoundingClientRect();
                   if (rect.width === 0 || rect.height === 0) return;
                   let altText = element.getAttribute('alt') || '';
                   let nameText = element.getAttribute('name') || '';
                   altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                   nameText = nameText.replace(/"/g, '');
                   const classList = element.className || '';
                   const id = element.id || '';
                   if (element.tagName.toLowerCase() === 'select') {{
                       // For select elements, get the text content of each option
                       textContent = Array.from(element.options).map(option => option.textContent).join('; ');
                   }} else if (element.tagName.toLowerCase() === 'img') {{
                       // For img elements, get the src attribute
                       imageUrl = element.src || '';
                       textContent = element.textContent || '';
                   }} else {{
                       // For other elements, get their text content
                       textContent = element.textContent || '';
                   }}
                   textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent
                   textContent = textContent.replace(/\\s+/g, ' ').trim(); // Remove extra whitespace
                   const titleText = element.getAttribute('title') || '';

                   // Determine if the element is interactable
                   const isInteractable = interactableSelectors.some(selector => element.matches(selector));
                 // Check if the current counter matches the element_id
                   if (counter == {element_id}) {{
                       if (nameText) {{
                           nameText += '{tag_str}';
                       }} else {{
                           nameText = '{tag_str}';
                       }}
                       element.setAttribute('name', nameText);
                   }}

                       // Calculate relative coordinates (x0, y0, x1, y1)
                   const relX0 = rect.left;
                   const relY0 = rect.top;
                   const relX1 = rect.right;
                   const relY1 = rect.bottom;

                   const dataString = [
                       counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                       (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                       (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                       altText, classList, id, textContent, isInteractable, titleText, imageUrl,
                       relX0, relY0, relX1, relY1 // Add the relative coordinates to the CSV line
                   ].map(value => `"${{value}}"`).join(",");


                   csvContent += dataString + "\\n";
                   counter++;
               }});

               return csvContent;
           }})();
           """
    # Save the bbox as a CSV
    csv_content = page.evaluate(js_script)

    return csv_content



def get_element_id(action, page: Page, data):
    # Given a raw traj_dict or any dict with key 'action' is the action taken
    # adds a '[SELECTED]' to the corresponding element on the page
    action_str = action.strip()
    action_name, infos = parse_action_str(action_str)
    csv_content = get_page_bboxes(page)

    element_id = None
    if len(infos) == 0:
        # these actions are not modifying any element
        pass
    elif len(infos) == 1:
        if action_name in ['click', 'clear', 'hover']:
            # info is the element_id for these actions
            element_id = int(infos[0])

        elif action_name == 'scroll':
            # info is direction
            pass
        elif action_name == 'press':
            # info is key combination
            pass
        elif action_name == 'goto':
            # info is url
            pass
        elif action_name == 'page_focus':
            # info is page number
            pass
        elif action_name == 'stop':
            # info is answer
            pass
        else:
            raise ValueError(f"Wrong action. The action {action_name} should not have 1 info")
    else:
        assert action_name == 'type'
        assert len(infos) == 2
        element_id = infos[0]
        text = infos[1]

    if element_id is not None:

        selected_id = draw_bounding_boxes(data_string=csv_content,
                                          screenshot_img=Image.open(BytesIO(page.screenshot())),
                                          select_idx=element_id,
                                          viewport_size={
                                                "width": data['viewport'][0],
                                                "height": data['viewport'][1]
                                            },
                                          offset=data['offset']
                                            )

        return True, element_id

    return False, action









    return page


def parse_action_str(action_str):
    # parse action string into action name and infos
    # info might contains
    action_name=action_str.split(' ')[0]
    action_infos=action_str[len(action_name)+1:]
    infos=[]
    l=len(action_infos)
    i=0
    while i<l:
        if action_infos[i]=='[':
            j=i+1
            content=''
            still_open=1
            while still_open>0:
                if action_infos[j]=='[':
                    still_open+=1
                elif action_infos[j]==']':
                    still_open-=1
                    if still_open==0:
                        break
                content+=action_infos[j]
                j+=1
            i=j+1
            infos.append(content)
        elif action_infos[i]==' ':
            pass
        else:
            raise ValueError('The action string is invalid')
        i+=1
    return action_name,infos




import numpy as np
import os
from playwright.sync_api import Page
from transfer_obs.processors import get_page_bboxes, draw_bounding_boxes
import pdb
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO, StringIO
def save_page_content(page, file_name: str = 'page_content.html'):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(page.content())
        print(f'页面内容已保存到: {file_name}')



def add_selected_tag(page, element_id, tag_str="[SELECTED_ELEMENT]") -> str:
    """JavaScript code to return bounding boxes and other metadata from HTML elements."""
    js_script = f"""
           (() => {{
               const interactableSelectors = [
                   'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                   '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                   '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                   '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'
               ];

               const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
               const modifiedTextSelectors = textSelectors.map(selector =>
                   `:not(${{interactableSelectors.join(', ')}}):not(style) > ${{selector}}`
               );

               const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
               const elements = document.querySelectorAll(combinedSelectors.join(', '));

               const pixelRatio = window.devicePixelRatio;
               let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable,Title,ImageUrl,RelX0,RelY0,RelX1,RelY1\\n";

               let counter = 1;
               let imageUrl = ''
               elements.forEach(element => {{
                   const rect = element.getBoundingClientRect();
                   if (rect.width === 0 || rect.height === 0) return;
                   let altText = element.getAttribute('alt') || '';
                   let nameText = element.getAttribute('name') || '';
                   altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                   nameText = nameText.replace(/"/g, '');
                   const classList = element.className || '';
                   const id = element.id || '';
                   if (element.tagName.toLowerCase() === 'select') {{
                       // For select elements, get the text content of each option
                       textContent = Array.from(element.options).map(option => option.textContent).join('; ');
                   }} else if (element.tagName.toLowerCase() === 'img') {{
                       // For img elements, get the src attribute
                       imageUrl = element.src || '';
                       textContent = element.textContent || '';
                   }} else {{
                       // For other elements, get their text content
                       textContent = element.textContent || '';
                   }}
                   textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent
                   textContent = textContent.replace(/\\s+/g, ' ').trim(); // Remove extra whitespace
                   const titleText = element.getAttribute('title') || '';

                   // Determine if the element is interactable
                   const isInteractable = interactableSelectors.some(selector => element.matches(selector));
                 // Check if the current counter matches the element_id
                   if (counter == {element_id}) {{
                       if (nameText) {{
                           nameText += '{tag_str}';
                       }} else {{
                           nameText = '{tag_str}';
                       }}
                       element.setAttribute('name', nameText);
                   }}

                       // Calculate relative coordinates (x0, y0, x1, y1)
                   const relX0 = rect.left;
                   const relY0 = rect.top;
                   const relX1 = rect.right;
                   const relY1 = rect.bottom;

                   const dataString = [
                       counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                       (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                       (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                       altText, classList, id, textContent, isInteractable, titleText, imageUrl,
                       relX0, relY0, relX1, relY1 // Add the relative coordinates to the CSV line
                   ].map(value => `"${{value}}"`).join(",");


                   csvContent += dataString + "\\n";
                   counter++;
               }});

               return csvContent;
           }})();
           """
    # Save the bbox as a CSV
    csv_content = page.evaluate(js_script)

    return csv_content



def get_element_id(action, page: Page, data):
    # Given a raw traj_dict or any dict with key 'action' is the action taken
    # adds a '[SELECTED]' to the corresponding element on the page
    action_str = action.strip()
    action_name, infos = parse_action_str(action_str)
    csv_content = get_page_bboxes(page)

    element_id = None
    if len(infos) == 0:
        # these actions are not modifying any element
        pass
    elif len(infos) == 1:
        if action_name in ['click', 'clear', 'hover']:
            # info is the element_id for these actions
            element_id = int(infos[0])

        elif action_name == 'scroll':
            # info is direction
            pass
        elif action_name == 'press':
            # info is key combination
            pass
        elif action_name == 'goto':
            # info is url
            pass
        elif action_name == 'page_focus':
            # info is page number
            pass
        elif action_name == 'stop':
            # info is answer
            pass
        else:
            raise ValueError(f"Wrong action. The action {action_name} should not have 1 info")
    else:
        assert action_name == 'type'
        assert len(infos) == 2
        element_id = infos[0]
        text = infos[1]

    if element_id is not None:

        selected_id = draw_bounding_boxes(data_string=csv_content,
                                          screenshot_img=Image.open(BytesIO(page.screenshot())),
                                          select_idx=element_id,
                                          viewport_size={
                                                "width": data['viewport'][0],
                                                "height": data['viewport'][1]
                                            },
                                          offset=data['offset']
                                            )

        return True, element_id

    return False, action









    return page


def parse_action_str(action_str):
    # parse action string into action name and infos
    # info might contains
    action_name=action_str.split(' ')[0]
    action_infos=action_str[len(action_name)+1:]
    infos=[]
    l=len(action_infos)
    i=0
    while i<l:
        if action_infos[i]=='[':
            j=i+1
            content=''
            still_open=1
            while still_open>0:
                if action_infos[j]=='[':
                    still_open+=1
                elif action_infos[j]==']':
                    still_open-=1
                    if still_open==0:
                        break
                content+=action_infos[j]
                j+=1
            i=j+1
            infos.append(content)
        elif action_infos[i]==' ':
            pass
        else:
            raise ValueError('The action string is invalid')
        i+=1
    return action_name,infos




