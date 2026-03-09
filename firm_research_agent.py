import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from hashlib import md5

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

# 配置日志
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置日志输出到文件和控制台
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'firm_research_agent.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("firm_research_agent")

# 配置缓存目录
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class RateLimiter:
    """速率限制器，用于控制API调用频率"""
    
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    def __call__(self):
        now = time.time()
        # 移除过期的调用记录
        while self.calls and self.calls[0] < now - self.period:
            self.calls.popleft()
        
        # 如果达到限制，等待
        if len(self.calls) >= self.max_calls:
            wait_time = self.period - (now - self.calls[0])
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
        
        # 记录新的调用
        self.calls.append(time.time())

# 创建全局速率限制器实例
serpapi_limiter = RateLimiter(max_calls=5, period=60)  # 每分钟最多5次调用
yfinance_limiter = RateLimiter(max_calls=10, period=60)  # 每分钟最多10次调用

def get_cache_key(query: str) -> str:
    """生成缓存键"""
    return md5(query.encode()).hexdigest()

def load_from_cache(cache_key: str):
    """从缓存加载数据"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        # 检查缓存是否过期（24小时）
        if time.time() - os.path.getmtime(cache_file) < 86400:
            with open(cache_file, "r", encoding="utf-8") as f:
                logger.info(f"Using cached data for key: {cache_key}")
                return json.load(f)
        else:
            logger.info(f"Cache expired for key: {cache_key}")
    return None

def save_to_cache(cache_key: str, data):
    """保存数据到缓存"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved data to cache for key: {cache_key}")

#!pip install yfinance matplotlib pytz numpy pandas python-dotenv requests bs4


def serpapi_search(query: str, num_results: int = 2, max_chars: int = 500, max_retries: int = 3) -> list:  # type: ignore[type-arg]
    """使用 SerpApi 进行 Google 搜索，支持超时和指数退避重试"""
    import re
    import requests
    from bs4 import BeautifulSoup

    # 检查缓存
    cache_key = get_cache_key(f"serpapi_{query}_{num_results}_{max_chars}")
    cached_data = load_from_cache(cache_key)
    if cached_data:
        logger.info(f"Using cached SerpAPI results for query: {query}")
        return cached_data

    # 应用速率限制
    serpapi_limiter()

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY not found in environment variables")

    url = "https://serpapi.com/search"
    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "num": num_results,
        "hl": "zh-cn",
        "gl": "cn",
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"SerpAPI attempt {attempt + 1}/{max_retries} for query: {query}")
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                wait_time = (2 ** attempt) * 2  # 增加等待时间
                logger.warning(f"SerpAPI rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                logger.error(f"SerpAPI error: {response.status_code}")
                raise Exception(f"Error in API request: {response.status_code}")

            data = response.json()

            if data.get("error"):
                wait_time = (2 ** attempt) * 2
                logger.warning(f"SerpAPI error: {data.get('error')}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            results = data.get("organic_results", [])
            break

        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.warning(f"SerpAPI timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                logger.warning(f"SerpAPI request error: {last_error}, retrying in {wait_time}s...")
                time.sleep(wait_time)
    else:
        logger.error(f"SerpAPI failed after {max_retries} attempts: {last_error}")
        raise Exception(f"SerpAPI failed after {max_retries} attempts: {last_error}")

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            content = ""
            for word in words:
                if len(content) + len(word) + 1 > max_chars:
                    break
                content += " " + word
            return content.strip()
        except Exception as e:
            logger.warning(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results[:num_results]:
        body = get_page_content(item.get("link", ""))
        enriched_results.append(
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "body": body,
            }
        )
        time.sleep(1)  # Be respectful to the servers

    # 保存到缓存
    save_to_cache(cache_key, enriched_results)
    
    logger.info(f"SerpAPI search completed successfully for query: {query}")
    return enriched_results


def analyze_stock(ticker: str) -> dict:  # type: ignore[type-arg]
    from datetime import datetime, timedelta

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from pytz import timezone  # type: ignore

    # 检查缓存
    cache_key = get_cache_key(f"stock_{ticker}")
    cached_data = load_from_cache(cache_key)
    if cached_data:
        logger.info(f"Using cached stock data for ticker: {ticker}")
        return cached_data

    # 应用速率限制
    yfinance_limiter()

    logger.info(f"Analyzing stock data for ticker: {ticker}")
    stock = yf.Ticker(ticker)

    # Get historical data (1 year of data to ensure we have enough for 200-day MA)
    end_date = datetime.now(timezone("UTC"))
    start_date = end_date - timedelta(days=365)
    
    try:
        hist = stock.history(start=start_date, end=end_date)
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}: {str(e)}")
        # 返回错误信息，建议使用搜索功能
        return {
            "error": f"无法获取 {ticker} 的股票数据。API速率限制或网络问题。请使用搜索功能获取最新的股票价格信息。",
            "ticker": ticker,
        }

    # Ensure we have data
    if hist.empty:
        logger.warning(f"No historical data available for {ticker}")
        return {
            "error": f"No historical data available for {ticker}. Please search for current stock price using SerpApi.",
            "ticker": ticker,
        }

    # Compute basic statistics and additional metrics
    current_price = stock.info.get("currentPrice", hist["Close"].iloc[-1])
    year_high = stock.info.get("fiftyTwoWeekHigh", hist["High"].max())
    year_low = stock.info.get("fiftyTwoWeekLow", hist["Low"].min())

    # Calculate 50-day and 200-day moving averages
    ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
    ma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]

    # Calculate YTD price change and percent change
    ytd_start = datetime(end_date.year, 1, 1, tzinfo=timezone("UTC"))
    ytd_data = hist.loc[ytd_start:]  # type: ignore[misc]
    if not ytd_data.empty:
        price_change = ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0]
        percent_change = (price_change / ytd_data["Close"].iloc[0]) * 100
    else:
        price_change = percent_change = np.nan

    # Determine trend
    if pd.notna(ma_50) and pd.notna(ma_200):
        if ma_50 > ma_200:
            trend = "Upward"
        elif ma_50 < ma_200:
            trend = "Downward"
        else:
            trend = "Neutral"
    else:
        trend = "Insufficient data for trend analysis"

    # Calculate volatility (standard deviation of daily returns)
    daily_returns = hist["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility

    # Create result dictionary
    result = {
        "ticker": ticker,
        "current_price": current_price,
        "52_week_high": year_high,
        "52_week_low": year_low,
        "50_day_ma": ma_50,
        "200_day_ma": ma_200,
        "ytd_price_change": price_change,
        "ytd_percent_change": percent_change,
        "trend": trend,
        "volatility": volatility,
    }

    # Convert numpy types to Python native types for better JSON serialization
    for key, value in result.items():
        if isinstance(value, np.generic):
            result[key] = value.item()

    # Generate plot
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist["Close"], label="Close Price")
    plt.plot(hist.index, hist["Close"].rolling(window=50).mean(), label="50-day MA")
    plt.plot(hist.index, hist["Close"].rolling(window=200).mean(), label="200-day MA")
    plt.title(f"{ticker} Stock Price (Past Year)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Save plot to file
    os.makedirs("coding", exist_ok=True)
    plot_file_path = f"coding/{ticker}_stockprice.png"
    plt.savefig(plot_file_path)
    plt.close()
    logger.info(f"Plot saved as {plot_file_path}")
    result["plot_file_path"] = plot_file_path

    # 保存到缓存
    save_to_cache(cache_key, result)
    
    logger.info(f"Stock analysis completed successfully for ticker: {ticker}")
    return result


def save_report_to_file(content: str, filename: str = "financial_report.md") -> str:
    """保存报告到 Markdown 文件"""
    import re
    from datetime import datetime

    output_dir = "coding"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    logger.info(f"Saving report to {filepath}")

    # 在报告末尾添加生成时间，不修改报告内容
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    # 检查是否已有生成时间标记，如果有则更新，没有则添加
    if "*报告生成时间：" in content:
        # 使用正则表达式匹配并替换生成时间
        content = re.sub(
            r'\*报告生成时间：.*?\*', 
            f'*报告生成时间：{current_date}*', 
            content
        )
    else:
        # 在报告末尾添加生成时间
        content += f"\n\n---\n\n*报告生成时间：{current_date}*"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Report saved successfully to {filepath}")
    return filepath


async def main():
    logger.info("Starting firm research agent")
    
    serpapi_search_tool = FunctionTool(
        serpapi_search, description="使用SerpApi搜索Google，返回包含摘要和正文内容的结果"
    )
    stock_analysis_tool = FunctionTool(
        analyze_stock, description="分析股票数据并生成图表"
    )
    save_report_tool = FunctionTool(
        save_report_to_file, description="将财务报告保存到markdown文件"
    )

    # 使用环境变量配置模型客户端
    logger.info("Configuring model client")
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    logger.info("Creating agents")
    search_agent = AssistantAgent(
        name="SerpApi_Search_Agent",
        model_client=model_client,
        tools=[serpapi_search_tool],
        description="使用SerpApi搜索Google，返回前2个结果，包含摘要和正文内容",
        system_message="你是一个有帮助的AI助手，使用你的工具解决任务。请用中文回复。",
    )

    stock_analysis_agent = AssistantAgent(
        name="Stock_Analysis_Agent",
        model_client=model_client,
        tools=[stock_analysis_tool],
        description="分析股票数据并生成图表",
        system_message="执行数据分析。请用中文回复。",
    )

    report_agent = AssistantAgent(
        name="Report_Agent",
        model_client=model_client,
        tools=[save_report_tool],
        description="基于搜索和股票分析结果生成报告",
        system_message="你是一个有帮助的助手，能够基于搜索和股票分析结果生成关于给定主题的综合报告。完成报告后，保存到文件并回复TERMINATE。请用中文生成报告。",
    )

    task = sys.argv[1] if len(sys.argv) > 1 else "分析苹果公司的财务报告"
    # Add current date to the task to ensure the report uses the correct date
    from datetime import datetime
    current_date = datetime.now().strftime("%Y年%m月%d日")
    task_with_date = f"{task}。注意：当前日期是{current_date}，请在报告中使用正确的日期，并使用中文生成报告。"
    
    logger.info(f"Task: {task_with_date}")
    team = RoundRobinGroupChat([stock_analysis_agent, search_agent, report_agent], max_turns=5)

    logger.info("Starting team execution")
    stream = team.run_stream(task=task_with_date)
    await Console(stream)

    logger.info("Closing model client")
    await model_client.close()
    logger.info("Firm research agent completed")


if __name__ == "__main__":
    asyncio.run(main())

