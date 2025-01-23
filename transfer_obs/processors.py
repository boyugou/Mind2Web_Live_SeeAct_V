import pandas as pd
import pdb
from io import BytesIO, StringIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import re
def rectangles_overlap(rect1, rect2, padding):
    """
    Check if two rectangles overlap.
    Each rectangle is represented as a list [x1, y1, x2, y2].
    """
    return not (
            rect1[2] < rect2[0] + padding
            or rect1[0] > rect2[2] - padding
            or rect1[1] > rect2[3] - padding
            or rect1[3] < rect2[1] + padding
    )
def get_page_bboxes(page) -> str:
    """JavaScript code to return bounding boxes and other metadata from HTML elements."""
    js_script = """
       (() => {
           const interactableSelectors = [
               'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
               '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
               '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
               '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'
           ];

           const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
           const modifiedTextSelectors = textSelectors.map(selector =>
               `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
           );

           const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
           const elements = document.querySelectorAll(combinedSelectors.join(', '));

           const pixelRatio = window.devicePixelRatio;
           let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable,Title,ImageUrl,RelX0,RelY0,RelX1,RelY1\\n";

           let counter = 1;
           let imageUrl = ''
           elements.forEach(element => {
               const rect = element.getBoundingClientRect();
               if (rect.width === 0 || rect.height === 0) return;
               let altText = element.getAttribute('alt') || '';
               altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
               const classList = element.className || '';
               const id = element.id || '';
               if (element.tagName.toLowerCase() === 'select') {
                   // For select elements, get the text content of each option
                   textContent = Array.from(element.options).map(option => option.textContent).join('; ');
               } else if (element.tagName.toLowerCase() === 'img') {
                   // For img elements, get the src attribute
                   imageUrl = element.src || '';
                   textContent = element.textContent || '';
               } else {
                   // For other elements, get their text content
                   textContent = element.textContent || '';
               }
               textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent
               textContent = textContent.replace(/\\s+/g, ' ').trim(); // Remove extra whitespace
               const titleText = element.getAttribute('title') || '';

               // Determine if the element is interactable
               const isInteractable = interactableSelectors.some(selector => element.matches(selector));


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
               ].map(value => `"${value}"`).join(",");


               csvContent += dataString + "\\n";
               counter++;
           });

           return csvContent;
       })();
       """
    # Save the bbox as a CSV
    csv_content = page.evaluate(js_script)

    return csv_content

def draw_bounding_boxes(
        data_string,
        screenshot_img,
        select_idx,
        viewport_size=None,
        offset=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
        add_text=True
):
    """
    min_width and min_height: Minimum dimensions of the bounding box to be plotted.
    """
    # Read CSV data
    if isinstance(data_string, str):
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
    else:
        df = data_string
    df["Area"] = df["Width"] * df["Height"]

    def update_text_content(row):
        if pd.notna(row["Title"]):
            if isinstance(row['Title'], str):
                title = row['Title'].strip()
                if title.strip() == '':
                    return row["TextContent"]
            else:

                title = str(row['Title']).strip()
                if title == '':
                    return row["TextContent"]
            return f'{row["TextContent"]} (Title: {row["Title"]})'
        return row["TextContent"]

    df["TextContent"] = df.apply(update_text_content, axis=1)

    # Remove bounding boxes that are clipped.
    b_x, b_y = (
        offset[0],
        offset[1],
    )
    if viewport_size is not None:
        df = df[
            (df["Bottom"] - b_y >= 0)
            & (df["Top"] - b_y <= viewport_size["height"])
            & (df["Right"] - b_x >= 0)
            & (df["Left"] - b_x <= viewport_size["width"])
            ]
        viewport_area = viewport_size["width"] * viewport_size["height"]
        # Filter out bounding boxes that too large (more than 80% of the viewport)
        df = df[df["Area"] <= 0.8 * viewport_area]

    # Open the screenshot image
    img = screenshot_img.copy()
    draw = ImageDraw.Draw(img)

    # Load a TTF font with a larger size
    font_path = "media/SourceCodePro-SemiBold.ttf"
    font_size, padding = 16, 2
    font = ImageFont.truetype(font_path, font_size)

    # Create a color cycle using one of the categorical color palettes in matplotlib
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bbox_id2visid = {}
    bbox_id2desc = {}
    index = 0
    id2center = {}
    existing_text_rectangles = []
    text_to_draw = []
    # Provide [id] textContent inputs to the model as text.
    text_content_elements = []
    text_content_text = set()  # Store text of interactable elements

    # Iterate through each row in the CSV and draw bounding boxes\

    def get_heading_level(tag):
        match = re.match(r'^H([1-6])$', tag)
        if match:
            return int(match.group(1))
        return None

    def generate_stars(n):
        return '*' * n

    for _, row in df.iterrows():

        if not row["Interactable"]:
            if not add_text:
                continue
            content = ""

            coordinate = [str(int(row['RelX0'])), str(int(row['RelY0'])),
                          str(int(row['RelX1'])), str(int(row['RelY1']))]
            coordinate = ", ".join(coordinate)
            content += f"[{coordinate}] "
            # Add image alt-text to the text representation.
            if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                content += row["Alt"]
            # Add HTML textContent (if any) to the text representation.
            h_level = get_heading_level(row["Element"])
            if h_level is not None:
                level_star = generate_stars(h_level)
                content += f"{level_star}Heading{h_level}: "
            if pd.notna(row["TextContent"]):
                content += (
                               row["TextContent"].strip().replace("\n", "").replace("\t", "")
                           )[
                           :200
                           ]  # Limit to 200 characters to avoid having too much text

            # Check if the text is a CSS selector
            if content and not (content.startswith(".") and "{" in content):
                # Add elements which are not interactable as StaticText
                if content not in text_content_text:
                    text_content_elements.append(f"[] [StaticText] [{content}]")
                    text_content_text.add(content)
            continue

        if (plot_ids is not None) and (row["ID"] not in plot_ids):
            continue

        unique_id = str(index + 1)
        bbox_id2visid[row["ID"]] = (
            unique_id  # map the bounding box ID to the unique character ID
        )
        top, right, bottom, left, width, height = (
            row["Top"],
            row["Right"],
            row["Bottom"],
            row["Left"],
            row["Width"],
            row["Height"],
        )
        left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y
        id2center[unique_id] = (
            (left + right) / 2,
            (bottom + top) / 2,
            width,
            height,
        )

        if width >= min_width and height >= min_height:
            # Get the next color in the cycle
            color = bbox_color or color_cycle[index % len(color_cycle)]
            draw.rectangle(
                [
                    left - bbox_padding,
                    top - bbox_padding,
                    right + bbox_padding,
                    bottom + bbox_padding,
                ],
                outline=color,
                width=bbox_border,
            )
            bbox_id2desc[row["ID"]] = color

            # Draw the text on top of the rectangle
            if add_ids:
                # Calculate list of possible text positions
                text_positions = [
                    (left - font_size, top - font_size),  # Top-left corner
                    (
                        left,
                        top - font_size,
                    ),  # A little to the right of the top-left corner
                    (right, top - font_size),  # Top-right corner
                    (
                        right - font_size - 2 * padding,
                        top - font_size,
                    ),  # A little to the left of the top-right corner
                    (left - font_size, bottom),  # Bottom-left corner
                    (
                        left,
                        bottom,
                    ),  # A little to the right of the bottom-left corner
                    (
                        right - font_size - 2 * padding,
                        bottom,
                    ),  # A little to the left of the bottom-right corner
                    (
                        left,
                        bottom,
                    ),  # A little to the right of the bottom-left corner
                    (
                        right - font_size - 2 * padding,
                        bottom,
                    ),  # A little to the left of the bottom-right corner
                ]
                text_width = draw.textlength(unique_id, font=font)
                text_height = font_size  # Assume the text is one line

                if viewport_size is not None:
                    for text_position in text_positions:
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                        # Check if the new text rectangle is within the viewport
                        if (
                                new_text_rectangle[0] >= 0
                                and new_text_rectangle[1] >= 0
                                and new_text_rectangle[2] <= viewport_size["width"]
                                and new_text_rectangle[3] <= viewport_size["height"]
                        ):
                            # If the rectangle is within the viewport, check for overlaps
                            overlaps = False
                            for existing_rectangle in existing_text_rectangles:
                                if rectangles_overlap(
                                        new_text_rectangle,
                                        existing_rectangle,
                                        padding * 2,
                                ):
                                    overlaps = True
                                    break

                            if not overlaps:
                                break
                        else:
                            # If the rectangle is outside the viewport, try the next position
                            continue
                else:
                    # If none of the corners work, move the text rectangle by a fixed amount
                    text_position = (
                        text_positions[0][0] + padding,
                        text_positions[0][1],
                    )
                    new_text_rectangle = [
                        text_position[0] - padding,
                        text_position[1] - padding,
                        text_position[0] + text_width + padding,
                        text_position[1] + text_height + padding,
                    ]

                existing_text_rectangles.append(new_text_rectangle)
                text_to_draw.append(
                    (new_text_rectangle, text_position, unique_id, color)
                )

                content = ""
                coordinate = [str(int(row['RelX0'])), str(int(row['RelY0'])),
                              str(int(row['RelX1'])), str(int(row['RelY1']))]
                coordinate = ", ".join(coordinate)
                content += f"[{coordinate}] "
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                    tmp_url = row["ImageUrl"]
                    content += f"[IMG_URL_ANNO]{tmp_url}"
                if pd.notna(row["TextContent"]):
                    content += (
                                   row["TextContent"]
                                   .strip()
                                   .replace("\n", "")
                                   .replace("\t", "")
                               )[
                               :200
                               ]  # Limit to 200 characters
                text_content_elements.append(
                    f"[{unique_id}] [{row['Element']}] [{content}]"
                )
                if int(unique_id) == int(select_idx):
                    return int(row["ID"])
                if content in text_content_text:
                    # Remove text_content_elements with content
                    text_content_elements = [
                        element
                        for element in text_content_elements
                        if element.strip() != content
                    ]
                text_content_text.add(content)

        index += 1

    for text_rectangle, text_position, unique_id, color in text_to_draw:
        # Draw a background rectangle for the text
        draw.rectangle(text_rectangle, fill=color)
        draw.text(text_position, unique_id, font=font, fill="white")

    content_str = "\n".join(text_content_elements)
    return img, id2center, content_str