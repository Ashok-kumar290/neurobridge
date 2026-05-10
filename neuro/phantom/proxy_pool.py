import httpx
import random
import time
import logging
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger(__name__)

class ProxyPool:
    """
    Maintains a dynamic pool of clear-net HTTP/HTTPS proxies.
    Automatically scrapes new proxies when the pool runs dry, tests them,
    and provides a robust IP-rotation layer for stealth data extraction.
    """
    def __init__(self, source_url: str = "https://free-proxy-list.net/"):
        self.source_url = source_url
        self.proxies = []
        self.refresh_pool()

    def refresh_pool(self):
        """Scrape the indexer for fresh proxies."""
        logger.info(f"[ProxyPool] Scraping clear-net indexer: {self.source_url}")
        try:
            response = httpx.get(self.source_url, timeout=10.0)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', attrs={'class': 'table table-striped table-bordered'})
            if not table:
                logger.error("[ProxyPool] Failed to find proxy table structure.")
                return
                
            new_proxies = []
            for row in table.find('tbody').find_all('tr'):
                cols = row.find_all('td')
                # Filter for HTTPS and Elite/Anonymous to ensure true IP masking
                if cols[6].text == 'yes': 
                    ip = cols[0].text
                    port = cols[1].text
                    new_proxies.append(f"http://{ip}:{port}")
            
            self.proxies = new_proxies
            logger.info(f"[ProxyPool] Successfully loaded {len(self.proxies)} fresh HTTPS proxies.")
        except Exception as e:
            logger.error(f"[ProxyPool] Failed to refresh pool: {e}")

    def get_proxy(self) -> Optional[str]:
        """Returns a random proxy from the pool, refreshing if empty."""
        if not self.proxies:
            self.refresh_pool()
            
        if self.proxies:
            return random.choice(self.proxies)
        return None

    def remove_proxy(self, proxy_url: str):
        """Removes a dead proxy from the active pool."""
        if proxy_url in self.proxies:
            self.proxies.remove(proxy_url)
            logger.debug(f"[ProxyPool] Evicted dead node: {proxy_url}")

    def stealth_get(self, url: str, max_retries: int = 5) -> httpx.Response:
        """
        Executes a GET request using continuous IP rotation.
        If a proxy fails, it is evicted and a new one is drawn until success.
        """
        for attempt in range(1, max_retries + 1):
            proxy = self.get_proxy()
            if not proxy:
                raise ConnectionError("Proxy pool is completely empty.")
                
            logger.info(f"[Stealth Scrape] Attempt {attempt}/{max_retries} via Node: {proxy}")
            
            transport = httpx.HTTPTransport(proxy=proxy)
            client = httpx.Client(transport=transport, timeout=8.0)
            
            try:
                response = client.get(url)
                response.raise_for_status()
                return response
            except Exception as e:
                logger.warning(f"[Stealth Scrape] Node failed ({type(e).__name__}). Rotating IP...")
                self.remove_proxy(proxy)
                
        raise ConnectionError(f"Failed to scrape {url} after {max_retries} proxy hops.")

if __name__ == "__main__":
    # Configure logging for standalone test
    logging.basicConfig(level=logging.INFO)
    pool = ProxyPool()
    
    # Test a heavily rate-limited endpoint
    try:
        resp = pool.stealth_get("https://httpbin.org/ip")
        print(f"Success! Response IP: {resp.json().get('origin')}")
    except Exception as e:
        print(f"Final failure: {e}")
