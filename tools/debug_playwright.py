import logging
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.DEBUG)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    try:
        page.goto(f"http://127.0.0.1:9980//index.php?page=login", timeout=60000)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("success")
        browser.close()
