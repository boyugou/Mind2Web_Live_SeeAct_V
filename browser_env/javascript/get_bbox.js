() => {
    const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'table'];
    const pixelRatio = window.devicePixelRatio;
    let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Id,TextContent,Interactable\n";
    let counter = 1;

    const salientAttributes = [
        "alt",
        "aria-describedby",
        "aria-label",
        "aria-role",
        "input-checked",
        // "input-value",
        "label",
        "name",
        "option_selected",
        "placeholder",
        "readonly",
        "text-value",
        "title",
        "value",
        "data-vote-target"
    ];

    const elements = Array.prototype.slice.call(document.querySelectorAll('*')).map(function(element) {
        const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);

        const rects = [...element.getClientRects()].filter(bb => {
            const center_x = bb.left + bb.width / 2;
            const center_y = bb.top + bb.height / 2;
            const elAtCenter = document.elementFromPoint(center_x, center_y);

            if (!elAtCenter) return false;
            return elAtCenter === element || element.contains(elAtCenter);
        }).map(bb => {
            const rect = {
                left: Math.max(0, bb.left),
                top: Math.max(0, bb.top),
                right: Math.min(vw, bb.right),
                bottom: Math.min(vh, bb.bottom)
            };
            return {
                ...rect,
                width: rect.right - rect.left,
                height: rect.bottom - rect.top
            };
        });

        const area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);

        const tagName = element.tagName.toLowerCase?.() || "";

        const nonClickableTags = [
            "label",
            "legend",
            "tr",
            "th",
            "td",
            "table",
             'p', 'span', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'
        ];
        let isClickable = false;


        if (tagName === "img") {
            let mapName = element.getAttribute("usemap");
            if (mapName) {
                const imgClientRects = element.getClientRects();
                mapName = mapName.replace(/^#/, "").replace('"', '\\"');
                const map = document.querySelector(`map[name=\"${mapName}\"]`);
                if (map && (imgClientRects.length > 0)) isClickable = true;
            }
        }

        if (!isClickable) {
            const role = element.getAttribute("role");
            const clickableRoles = [
                "button", "tab", "link", "checkbox", "menuitem", "menuitemcheckbox", "menuitemradio", "radio"
            ];
            if (role != null && clickableRoles.includes(role.toLowerCase())) {
                isClickable = true;
            } else {
                const contentEditable = element.getAttribute("contentEditable");
                if (contentEditable != null && ["", "contenteditable", "true"].includes(contentEditable.toLowerCase())) {
                    isClickable = true;
                }
            }
        }

        if (!isClickable && element.hasAttribute("jsaction")) {
            const jsactionRules = element.getAttribute("jsaction").split(";");
            for (let jsactionRule of jsactionRules) {
                const ruleSplit = jsactionRule.trim().split(":");
                if ((ruleSplit.length >= 1) && (ruleSplit.length <= 2)) {
                    const [eventType, namespace, actionName] = ruleSplit.length === 1
                        ? ["click", ...ruleSplit[0].trim().split("."), "_"]
                        : [ruleSplit[0], ...ruleSplit[1].trim().split("."), "_"];
                    if (!isClickable) {
                        isClickable = (eventType === "click") && (namespace !== "none") && (actionName !== "_");
                    }
                }
            }
        }

        if (!isClickable) {
            const clickableTags = [
                "input", "textarea", "select", "button", "a", "iframe", "video", "object", "embed", "details"
            ];
            isClickable = clickableTags.includes(tagName);
        }

        if (!isClickable) {
            if (tagName === "label")
                isClickable = (element.control != null) && !element.control.disabled;
            else if (tagName === "img")
                isClickable = ["zoom-in", "zoom-out"].includes(element.style.cursor);
        }

        const className = element.getAttribute("class");
        if (!isClickable && className && className.toLowerCase().includes("button")) {
            isClickable = true;
        }

        const tabIndexValue = element.getAttribute("tabindex");
        const tabIndex = tabIndexValue ? parseInt(tabIndexValue) : -1;
        if (!isClickable && !(tabIndex < 0) && !isNaN(tabIndex)) {
            isClickable = true;
        }


        return {
            element: element,
            include: isClickable,
            area,
            rects,
            text: element.textContent.trim().replace(/\s{2,}/g, ' '),
            isInteractable: isClickable
        };
    });

    // Split elements into clickable and text elements
    const clickableElements = elements.filter(item => item.include && (item.area >= 1));
    const textElements = elements.filter(item => !item.include && textSelectors.some(selector => item.element.matches(selector)));

    // Filter clickable elements to remove nested ones
    const filteredClickableElements = clickableElements.filter(x =>
        !clickableElements.some(y => x.element.contains(y.element) && !(x == y))
    );

    // Function to process table elements and avoid duplicates
    const processTableElement = (element) => {
        let tableText = '| ';
        const rows = element.querySelectorAll('tr');
        rows.forEach((row, rowIndex) => {
            const cells = row.querySelectorAll('th, td');
            let rowText = '';
            cells.forEach(cell => {
                rowText += cell.textContent.trim().replace(/\s{2,}/g, ' ') + ' | ';
            });
            if (rowIndex === 0) {
                tableText += rowText.replace(/\| $/, '') + ' |\n';
                tableText += '| ' + '- | '.repeat(cells.length) + '\n';
            } else {
                tableText += '| ' + rowText.replace(/\| $/, '') + ' |\n';
            }
        });
        return tableText.trim();
    };

    // Function to check if an element is a child of another element in the list
    const isChildOfAny = (element, elements) => {
        return elements.some(parent => parent.element.contains(element) && parent.element !== element);
    };

    // Filter text elements to remove nested ones
    const filteredTextElements = textElements.filter(x =>
        !isChildOfAny(x.element, textElements)
    );

    // Merge elements while maintaining their original order
    const finalElements = elements.map(item => {
        if (filteredClickableElements.includes(item)) {
            return filteredClickableElements.find(clickable => clickable.element === item.element);
        }
        if (filteredTextElements.includes(item)) {
            return filteredTextElements.find(text => text.element === item.element);
        }
        return null;
    }).filter(item => item != null);

    // Function to recursively extract text content with \n between elements


    const skipTags = [
        "input",
        "textarea",
        "select"
    ];

    const extractTextContent = (element) => {
        let texts = [];

        element.childNodes.forEach(child => {
            if (child.nodeType === Node.TEXT_NODE && child.textContent.trim()) {
                texts.push(child.textContent.trim());
            } else if (child.nodeType === Node.ELEMENT_NODE) {
                // Skip elements in the skipTags list
                if (skipTags.includes(child.tagName.toLowerCase())) {
                    return;
                }

                let attributesText = salientAttributes
                    .filter(attr => child.hasAttribute(attr))
                    .map(attr => `${child.getAttribute(attr)}`)
                    .join(', ');

                const childText = extractTextContent(child);
                if (attributesText) {
                    texts.push(`${attributesText}: ${childText}`);
                } else if (childText) {
                    texts.push(childText);
                }
            }
        });

        return texts.join('\n'); // 使用 '\n' 连接子节点的文本
    };


    finalElements.forEach(item => {
        const element = item.element;
        const rect = element.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        let altText = element.getAttribute('alt') || '';
        altText = altText.replace(/"/g, '""');
        const id = element.id || '';
        let textContent = '';

        if (element.tagName.toLowerCase() === 'select') {
            textContent = "Options: ";
            textContent += Array.from(element.options).map(option => option.textContent).join('; ');
        } else if (element.tagName.toLowerCase() === 'table') {
            textContent = processTableElement(element);
        } else {
            textContent = extractTextContent(element);
        }

        salientAttributes.forEach(attr => {
            const attrValue = element.getAttribute(attr);
            if (attrValue) {
                textContent += `${attrValue}.`;
            }
        });

        textContent = textContent.replace(/"/g, '""');
        textContent = textContent.replace(/[ \t]+/g, ' ');

        const dataString = [
            counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
            (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
            (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
            altText, id, textContent, item.isInteractable
        ].map(value => `"${value}"`).join(",");

        csvContent += dataString + "\n";
        counter++;
    });

    return csvContent;
}
