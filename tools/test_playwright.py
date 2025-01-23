from playwright.sync_api import sync_playwright
CLASSIFIEDS_port = "9980"
CLASSIFIEDS = f"http://127.0.0.1:{CLASSIFIEDS_port}"

context_manager = sync_playwright()
playwright = context_manager.__enter__()
browser = playwright.chromium.launch(headless=True)
context = browser.new_context()
page = context.new_page()
page.goto(CLASSIFIEDS)
print("successfull connection")
page.goto(f"{CLASSIFIEDS}/index.php?page=login")