"""Web browser tool — full rendering via Playwright.

Renders JavaScript-heavy sites, takes screenshots, and extracts clean text.
"""

from pathlib import Path

from playwright.async_api import async_playwright
import structlog

log = structlog.get_logger()


class WebBrowserTool:
    def __init__(self, screenshot_dir: Path | None = None):
        self.screenshot_dir = screenshot_dir

    async def fetch(self, url: str, screenshot: bool = True) -> dict:
        """Render a URL using Playwright and return simplified text content."""
        if not url.startswith("http"):
            url = f"https://{url}"

        try:
            log.info("web_browser_fetching", url=url)
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Set a common user agent
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={'width': 1280, 'height': 800}
                )
                page = await context.new_page()
                
                # Wait for load and some idle time to let JS run
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception as e:
                    log.debug("web_browser_networkidle_timeout", url=url, error=str(e))
                    await page.goto(url, wait_until="load", timeout=20000)
                
                # Scroll to bottom and back to top to trigger any lazy loading
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                await page.evaluate("window.scrollTo(0, 0)")
                await page.wait_for_timeout(1000)
                
                # Get page title
                title = await page.title()

                # Take screenshot before we strip elements for text extraction
                screenshot_path = None
                if screenshot and self.screenshot_dir:
                    self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                    # Sanitize URL into a filename
                    safe_name = url.split("//")[-1].replace("/", "_").replace("?", "_")[:80]
                    screenshot_path = self.screenshot_dir / f"{safe_name}.png"
                    await page.screenshot(path=str(screenshot_path), full_page=True)
                    log.info("web_browser_screenshot", path=str(screenshot_path))
                
                # Extract clean text via evaluating in-page script
                # We're going back to a simpler but more reliable innerText approach
                # but we'll also look into Shadow DOMs if they exist.
                text = await page.evaluate('''() => {
                    function getDeepText(node) {
                        let text = "";
                        if (node.nodeType === Node.TEXT_NODE) {
                            text += node.textContent;
                        } else if (node.nodeType === Node.ELEMENT_NODE) {
                            // Skip hidden elements
                            const style = window.getComputedStyle(node);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return "";
                            }

                            // Handle Shadow DOM
                            if (node.shadowRoot) {
                                for (const child of node.shadowRoot.childNodes) {
                                    text += getDeepText(child);
                                }
                            }

                            // Handle light DOM
                            for (const child of node.childNodes) {
                                text += getDeepText(child);
                            }

                            // Add newlines for block elements
                            if (style.display === 'block' || style.display === 'flex' || ['P', 'DIV', 'BR', 'H1', 'H2', 'H3', 'LI'].includes(node.tagName)) {
                                text += "\\n";
                            }
                        }
                        return text;
                    }

                    // Remove noise before gathering text
                    const unwanted = ['script', 'style', 'iframe', 'svg', 'noscript', 'canvas'];
                    unwanted.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });

                    const deepText = getDeepText(document.body).trim();
                    return deepText || document.body.innerText;
                }''')

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                text = "\n".join(line for line in lines if line)

                log.info("web_browser_done", url=url, title=title, content_len=len(text))
                await browser.close()

                result = {
                    "url": url,
                    "title": title,
                    "content": text[:15000],  # Truncate for context window
                    "truncated": len(text) > 15000,
                }
                if screenshot_path and screenshot_path.exists():
                    result["screenshot"] = str(screenshot_path)
                return result
        except Exception as e:
            log.error("web_browser_failed", url=url, error=str(e))
            return {"error": str(e)}
