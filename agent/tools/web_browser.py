"""Web browser tool — full rendering and interaction via Playwright.

Supports:
  - fetch(url): Navigate and extract page content
  - click(selector): Click an element
  - type(selector, text): Type into an input field
  - search(query): Google search shortcut
  - screenshot(): Take a screenshot of the current page
  - get_links(): List all links on the current page

The browser session persists across calls within a single subagent run,
so a subagent can navigate to a page, click a link, fill a form, etc.
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
import structlog

log = structlog.get_logger()

# Auto-close the browser after this many seconds of inactivity
SESSION_TIMEOUT = 120


class WebBrowserTool:
    def __init__(self, screenshot_dir: Path | None = None):
        self.screenshot_dir = screenshot_dir
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_url: str = ""
        self._close_timer: asyncio.TimerHandle | None = None

    async def _ensure_browser(self):
        """Lazily start the browser and create a page."""
        if self._page and not self._page.is_closed():
            self._reset_timer()
            return

        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )
        self._page = await self._context.new_page()
        self._reset_timer()
        log.info("web_browser_session_started")

    def _reset_timer(self):
        """Reset the inactivity auto-close timer."""
        if self._close_timer:
            self._close_timer.cancel()
        try:
            loop = asyncio.get_running_loop()
            self._close_timer = loop.call_later(SESSION_TIMEOUT, self._schedule_close)
        except RuntimeError:
            pass

    def _schedule_close(self):
        asyncio.ensure_future(self.close())

    async def close(self):
        """Shut down the browser session."""
        if self._close_timer:
            self._close_timer.cancel()
            self._close_timer = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            self._page = None
            self._context = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        log.info("web_browser_session_closed")

    # ── Core actions ──────────────────────────────────────────

    async def fetch(self, url: str, screenshot: bool = True) -> dict:
        """Navigate to a URL and return the page content."""
        if not url.startswith("http"):
            url = f"https://{url}"

        try:
            await self._ensure_browser()
            log.info("web_browser_fetching", url=url)

            try:
                await self._page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception:
                await self._page.goto(url, wait_until="load", timeout=20000)

            self._last_url = url
            return await self._extract_page(screenshot=screenshot)

        except Exception as e:
            log.error("web_browser_failed", url=url, error=str(e))
            return {"error": str(e)}

    async def click(self, selector: str, screenshot: bool = True) -> dict:
        """Click an element on the current page.

        selector: CSS selector or text content (e.g. "text=Sign In", ".btn-primary",
                  "a:has-text('Next')", "#submit-button")
        """
        try:
            await self._ensure_browser()
            log.info("web_browser_click", selector=selector)

            # Try the selector directly first
            try:
                await self._page.click(selector, timeout=5000)
            except Exception:
                # Fall back to text-based selector
                if not selector.startswith("text="):
                    await self._page.click(f"text={selector}", timeout=5000)
                else:
                    raise

            # Wait for navigation or dynamic content
            try:
                await self._page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                await self._page.wait_for_timeout(2000)

            self._last_url = self._page.url
            return await self._extract_page(screenshot=screenshot)

        except Exception as e:
            log.error("web_browser_click_failed", selector=selector, error=str(e))
            return {"error": f"Click failed for '{selector}': {e}"}

    async def type_text(self, selector: str, text: str, submit: bool = False) -> dict:
        """Type text into an input field.

        selector: CSS selector for the input (e.g. "input[name='q']", "#search-box")
        text: the text to type
        submit: if True, press Enter after typing
        """
        try:
            await self._ensure_browser()
            log.info("web_browser_type", selector=selector, text_len=len(text))

            await self._page.fill(selector, text, timeout=5000)

            if submit:
                await self._page.press(selector, "Enter")
                try:
                    await self._page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    await self._page.wait_for_timeout(2000)

            self._last_url = self._page.url
            return await self._extract_page(screenshot=True)

        except Exception as e:
            log.error("web_browser_type_failed", selector=selector, error=str(e))
            return {"error": f"Type failed for '{selector}': {e}"}

    async def search(self, query: str) -> dict:
        """Perform a Google search and return the results page."""
        import urllib.parse
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        return await self.fetch(url)

    async def screenshot(self) -> dict:
        """Take a screenshot of the current page state."""
        try:
            await self._ensure_browser()
            path = await self._take_screenshot()
            result = {
                "url": self._page.url,
                "title": await self._page.title(),
            }
            if path:
                result["screenshot"] = str(path)
            return result
        except Exception as e:
            return {"error": f"Screenshot failed: {e}"}

    async def get_links(self, limit: int = 50) -> dict:
        """List clickable links on the current page with their selectors."""
        try:
            await self._ensure_browser()

            links = await self._page.evaluate(f'''() => {{
                const results = [];
                const seen = new Set();
                for (const el of document.querySelectorAll("a[href], button, [role='button'], [role='link']")) {{
                    const text = (el.innerText || el.textContent || "").trim().substring(0, 80);
                    const href = el.getAttribute("href") || "";
                    if (!text || seen.has(text)) continue;
                    seen.add(text);
                    results.push({{
                        text: text,
                        href: href,
                        tag: el.tagName.toLowerCase(),
                    }});
                    if (results.length >= {limit}) break;
                }}
                return results;
            }}''')

            return {
                "url": self._page.url,
                "title": await self._page.title(),
                "links": links,
            }
        except Exception as e:
            return {"error": f"Get links failed: {e}"}

    # ── Internal helpers ──────────────────────────────────────

    async def _extract_page(self, screenshot: bool = True) -> dict:
        """Extract text content from the current page."""
        title = await self._page.title()

        # Scroll to trigger lazy loading
        await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await self._page.wait_for_timeout(800)
        await self._page.evaluate("window.scrollTo(0, 0)")
        await self._page.wait_for_timeout(500)

        screenshot_path = None
        if screenshot:
            screenshot_path = await self._take_screenshot()

        text = await self._page.evaluate('''() => {
            function getDeepText(node) {
                let text = "";
                if (node.nodeType === Node.TEXT_NODE) {
                    text += node.textContent;
                } else if (node.nodeType === Node.ELEMENT_NODE) {
                    const style = window.getComputedStyle(node);
                    if (style.display === 'none' || style.visibility === 'hidden') {
                        return "";
                    }
                    if (node.shadowRoot) {
                        for (const child of node.shadowRoot.childNodes) {
                            text += getDeepText(child);
                        }
                    }
                    for (const child of node.childNodes) {
                        text += getDeepText(child);
                    }
                    if (style.display === 'block' || style.display === 'flex' ||
                        ['P', 'DIV', 'BR', 'H1', 'H2', 'H3', 'LI'].includes(node.tagName)) {
                        text += "\\n";
                    }
                }
                return text;
            }
            const unwanted = ['script', 'style', 'iframe', 'svg', 'noscript', 'canvas'];
            unwanted.forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
            const deepText = getDeepText(document.body).trim();
            return deepText || document.body.innerText;
        }''')

        lines = (line.strip() for line in text.splitlines())
        text = "\n".join(line for line in lines if line)

        log.info("web_browser_done", url=self._page.url, title=title, content_len=len(text))

        result = {
            "url": self._page.url,
            "title": title,
            "content": text[:15000],
            "truncated": len(text) > 15000,
        }
        if screenshot_path:
            result["screenshot"] = str(screenshot_path)
        return result

    async def _take_screenshot(self) -> Path | None:
        """Take a screenshot and return the file path."""
        if not self.screenshot_dir:
            return None
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._page.url.split("//")[-1].replace("/", "_").replace("?", "_")[:80]
        path = self.screenshot_dir / f"{safe_name}.png"
        await self._page.screenshot(path=str(path), full_page=True)
        log.info("web_browser_screenshot", path=str(path))
        return path
