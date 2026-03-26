"""Web fetch tool — simple async HTTP fetching with HTML to Markdown-ish text conversion.

Uses aiohttp and beautifulsoup4 to grab content and make it readable for LLMs.
"""

import aiohttp
from bs4 import BeautifulSoup
import structlog

log = structlog.get_logger()

class WebFetchTool:
    async def fetch(self, url: str) -> dict:
        """Fetch a URL and return simplified text content."""
        if not url.startswith("http"):
            url = f"https://{url}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        return {"error": f"HTTP {response.status} from {url}"}
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Remove script and style elements
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()

                    # Get text, clean up whitespace
                    text = soup.get_text(separator="\n")
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = "\n".join(chunk for chunk in chunks if chunk)

                    return {
                        "url": str(response.url),
                        "status": response.status,
                        "content": text[:15000],  # Truncate for context window
                        "truncated": len(text) > 15000
                    }
        except Exception as e:
            log.error("web_fetch_failed", url=url, error=str(e))
            return {"error": str(e)}
