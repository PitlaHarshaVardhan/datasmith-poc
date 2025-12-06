from duckduckgo_search import DDGS
from typing import List
import logging

logger = logging.getLogger(__name__)

def web_search(query: str, max_results: int = 5) -> List[str]:
    logger.info("Web search for query='%s'", query)
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "")
            href = r.get("href", "")
            snippet = r.get("body", "")
            results.append(f"{title} - {href}\n{snippet}")
    logger.info("Web search returned %d results", len(results))
    return results
