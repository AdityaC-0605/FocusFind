import os
import asyncio
import logging
import re
import bleach
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import aiohttp
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # Required API keys
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Application settings
    CACHE_TYPE = "SimpleCache"
    REQUEST_TIMEOUT = 10
    MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB
    MAX_SUMMARY_LENGTH = 15000
    RATE_LIMIT = "5 per minute"
    MAX_CONCURRENT_REQUESTS = 5
    ASYNC_TIMEOUT = 15

# Validate configuration
required_keys = [
    "GOOGLE_CSE_ID", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "OPENWEATHER_API_KEY", "ALPHA_VANTAGE_API_KEY", "NEWS_API_KEY"
]

for key in required_keys:
    if not getattr(Config, key):
        raise RuntimeError(f"Missing required configuration: {key}")

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
cache = Cache(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[app.config["RATE_LIMIT"]]
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure HTTP session with retries
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http_session = requests.Session()
http_session.mount("https://", adapter)
http_session.mount("http://", adapter)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=app.config["MAX_CONCURRENT_REQUESTS"])

# Utility functions
def sanitize_input(input_str, max_length=256):
    """Sanitize and validate input strings"""
    if not input_str or len(input_str) > max_length:
        raise ValueError("Invalid input length")
    return bleach.clean(input_str.strip())

def async_handler(f):
    """Decorator to handle async routes in Flask"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(f(*args, **kwargs))
            return result
        except Exception as e:
            logger.error(f"Async handler error: {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
        finally:
            loop.close()
    return wrapper

# API Clients
class APIClient:
    """Base API client with common functionality"""
    def __init__(self, base_url=None):
        self.session = http_session
        self.base_url = base_url

    def _request(self, method, endpoint, **kwargs):
        """Make an HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        try:
            response = self.session.request(
                method,
                url,
                timeout=app.config["REQUEST_TIMEOUT"],
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed to {url}: {str(e)}")
            raise

class GoogleClient(APIClient):
    """Google Custom Search API client"""
    def search(self, query):
        params = {
            "q": query,
            "cx": app.config["GOOGLE_CSE_ID"],
            "key": app.config["GOOGLE_API_KEY"],
            "num": 5
        }
        return self._request(
            "GET",
            "https://www.googleapis.com/customsearch/v1",
            params=params
        )

class GeminiClient(APIClient):
    """Gemini API client"""
    def generate_summary(self, text):
        if not text or len(text) > app.config["MAX_CONTENT_LENGTH"]:
            raise ValueError("Invalid content for summarization")
        
        payload = {
            "contents": [{
                "parts": [{"text": text[:app.config["MAX_SUMMARY_LENGTH"]]}]
            }]
        }
        
        return self._request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            params={"key": app.config["GEMINI_API_KEY"]},
            json=payload
        )

# Initialize API clients
google_client = GoogleClient()
gemini_client = GeminiClient()

# Web scraping
async def scrape_website(session, url):
    """Asynchronously scrape website content with headers"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=app.config["ASYNC_TIMEOUT"])
        ) as response:
            if response.status == 200:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                
                # Remove unwanted elements
                for element in soup(["script", "style", "iframe", "noscript"]):
                    element.decompose()
                
                # Extract and clean content
                content = " ".join([
                    p.text.strip() for p in soup.find_all("p") if p.text.strip()
                ])
                return bleach.clean(content, strip=True) or "No content found."
            return "Failed to retrieve content"
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return "Error scraping content"

# Routes
@app.route('/')
def index():
    return render_template("dummy.html")

@app.route('/search_summarize', methods=['GET'])
@limiter.limit(app.config["RATE_LIMIT"])
@cache.cached(timeout=3600, query_string=True)
@async_handler
async def search_and_summarize():
    """Search and summarize endpoint"""
    try:
        # Validate input
        raw_query = request.args.get("q", "")
        query = sanitize_input(raw_query)
        
        # Perform search
        results = google_client.search(query).get("items", [])
        if not results:
            return jsonify({"error": "No search results found"}), 404
            
        urls = [item["link"] for item in results[:5]]

        # Scrape websites
        async with aiohttp.ClientSession() as session:
            tasks = [scrape_website(session, url) for url in urls]
            contents = await asyncio.gather(*tasks)
            
        valid_contents = [
            c for c in contents 
            if c not in ["No content found.", "Error scraping content"]
        ]
        
        if not valid_contents:
            return jsonify({"error": "No valid content found"}), 404

        # Generate summaries
        summaries = []
        for content in valid_contents:
            try:
                result = gemini_client.generate_summary(content)
                summary = result["candidates"][0]["content"]["parts"][0]["text"]
                if len(summary) > 100:
                    summaries.append(summary[:500])
            except Exception as e:
                logger.warning(f"Summary generation failed for content: {str(e)}")
                continue

        if not summaries:
            return jsonify({"error": "Failed to generate summaries"}), 500

        # Generate final summary
        combined_text = " ".join(summaries)
        final_summary = gemini_client.generate_summary(
            f"Question: {query}\nContext: {combined_text}"
        )["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify({
            "query": query,
            "summaries": [{"url": url, "summary": summary} 
                         for url, summary in zip(urls, summaries)],
            "final_summary": final_summary
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Critical error in search_and_summarize: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/weather')
@limiter.limit(app.config["RATE_LIMIT"])
@cache.cached(timeout=600, query_string=True)
def get_weather():
    """Get current weather data"""
    try:
        location = sanitize_input(request.args.get("location", ""))
        
        response = http_session.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": location,
                "appid": app.config["OPENWEATHER_API_KEY"],
                "units": "metric"
            },
            timeout=app.config["REQUEST_TIMEOUT"]
        )
        response.raise_for_status()
        data = response.json()
        
        return jsonify({
            "location": data.get("name"),
            "temperature": data.get("main", {}).get("temp"),
            "description": data.get("weather", [{}])[0].get("description")
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API error: {str(e)}")
        return jsonify({"error": "Failed to fetch weather data"}), 502
    except Exception as e:
        logger.error(f"Unexpected weather error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/news')
@limiter.limit(app.config["RATE_LIMIT"])
@cache.cached(timeout=900, query_string=True)
def get_news():
    """Get news articles"""
    try:
        response = http_session.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "apiKey": app.config["NEWS_API_KEY"],
                "pageSize": 5,
                "country": "in"
            },
            timeout=app.config["REQUEST_TIMEOUT"]
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        return jsonify([{
            "title": a.get("title"),
            "url": a.get("url"),
            "description": a.get("description")
        } for a in articles[:3]])
    except requests.exceptions.RequestException as e:
        logger.error(f"News API error: {str(e)}")
        return jsonify({"error": "Failed to fetch news"}), 502
    except Exception as e:
        logger.error(f"Unexpected news error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stock/<symbol>')
@limiter.limit(app.config["RATE_LIMIT"])
@cache.cached(timeout=120, query_string=True)
def get_stock(symbol):
    """Get real-time stock data"""
    try:
        # Sanitize symbol input
        clean_symbol = re.sub(r"[^A-Z0-9.:-]", "", symbol.upper())
        if not clean_symbol:
            raise ValueError("Invalid stock symbol")
        
        response = http_session.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": clean_symbol,
                "apikey": app.config["ALPHA_VANTAGE_API_KEY"]
            },
            timeout=app.config["REQUEST_TIMEOUT"]
        )
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data:
            return jsonify({"error": "Invalid symbol or no data available"}), 404
            
        quote = data["Global Quote"]
        
        return jsonify({
            "symbol": clean_symbol,
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "percent_change": quote.get("10. change percent"),
            "volume": quote.get("06. volume"),
            "latest_trading_day": quote.get("07. latest trading day")
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.exceptions.RequestException as e:
        logger.error(f"Stock API error for {symbol}: {str(e)}")
        return jsonify({"error": "Failed to fetch stock data"}), 502
    except Exception as e:
        logger.error(f"Unexpected stock error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.teardown_appcontext
def teardown(exception=None):
    """Cleanup on application shutdown"""
    http_session.close()

if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", False),
        port=int(os.getenv("FLASK_PORT", 5001)),
        threaded=True
    )
