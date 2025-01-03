import requests
from bs4 import BeautifulSoup

class WS1:
    def __init__(self, ticker):
        """
        Initialize with a ticker symbol. Converts ticker to uppercase
        and constructs the FinViz URL.
        """
        self.ticker = str(ticker).upper()
        self.url = f"https://finviz.com/quote.ashx?t={self.ticker}&ty=c&ta=1&p=d"

    def scrape(self):
        """
        Scrapes FinViz for the provided ticker, returning a list of related symbols
        from the <span style="font-size:11px"> element (all <a> with class='tab-link').
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/106.0.0.0 Safari/537.36"
            )
        }
        try:
            resp = requests.get(self.url, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            print(f"Error requesting FinViz page: {e}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        span = soup.find("span", {"style": "font-size:11px"})
        if not span:
            return []

        # Within this span, find all <a class="tab-link">
        links = span.find_all("a", class_="tab-link")
        if not links:
            return []

        # Extract the text (symbol) from each link
        related_symbols = [link.get_text(strip=True) for link in links]
        return related_symbols
