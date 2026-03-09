"""
文献综述 Agent - 基于 arXiv 和 SerpApi 的自动化文献分析系统

功能：
1. 文献搜索与筛选（arXiv API + SerpApi 并行搜索）
2. 关键信息提取（方法、结论、贡献）
3. 趋势分析（研究热点、时间线）
4. 引用生成（BibTeX 格式）

架构：
- Search_Agent: 并行搜索 arXiv 和 SerpApi，合并结果
- Analysis_Agent: 提取关键信息、分析趋势
- Report_Agent: 生成结构化报告
"""

import os
import json
import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, List, Optional, Dict
from collections import defaultdict
from xml.etree import ElementTree
from urllib.parse import quote

import aiohttp
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from openai import RateLimitError

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'literature_review_agent.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("literature_review_agent")


class ArxivClient:
    """arXiv API 客户端"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[dict]:
        """
        搜索 arXiv 文献
        
        Args:
            query: 搜索关键词
            max_results: 最大结果数
            sort_by: 排序方式
            sort_order: 排序顺序
        
        Returns:
            文献列表
        """
        max_results = max_results or self.max_results
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        if self.session is None:
            logger.error("Session not initialized")
            return []
        
        try:
            logger.info(f"Searching arXiv: {self.BASE_URL}")
            async with self.session.get(self.BASE_URL, params=params, timeout=30) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    papers = self._parse_xml(xml_data)
                    if papers:
                        logger.info(f"Successfully fetched {len(papers)} papers from arXiv")
                        return papers
                else:
                    logger.error(f"arXiv API error: {response.status}")
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
        
        return []
    
    def _parse_xml(self, xml_data: str) -> List[dict]:
        """解析 arXiv XML 响应"""
        papers = []
        
        try:
            root = ElementTree.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', ns):
                id_elem = entry.find('atom:id', ns)
                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                published_elem = entry.find('atom:published', ns)
                updated_elem = entry.find('atom:updated', ns)
                
                if id_elem is None or title_elem is None:
                    continue
                
                id_text = id_elem.text if id_elem is not None else None
                title_text = title_elem.text if title_elem is not None else None
                summary_text = summary_elem.text if summary_elem is not None else None
                published_text = published_elem.text if published_elem is not None else None
                updated_text = updated_elem.text if updated_elem is not None else None
                
                paper = {
                    'id': id_text.split('/')[-1] if id_text else '',
                    'title': title_text.strip().replace('\n', ' ') if title_text else '',
                    'summary': summary_text.strip().replace('\n', ' ') if summary_text else '',
                    'published': published_text[:10] if published_text else '',
                    'updated': updated_text[:10] if updated_text else '',
                    'authors': [author.find('atom:name', ns).text 
                               for author in entry.findall('atom:author', ns)
                               if author.find('atom:name', ns) is not None and author.find('atom:name', ns).text],
                    'categories': [cat.get('term') for cat in entry.findall('atom:category', ns) if cat.get('term')],
                    'link': id_text if id_text else '',
                    'pdf_link': None
                }
                
                for link in entry.findall('atom:link', ns):
                    title_attr = link.get('title')
                    href_attr = link.get('href')
                    if title_attr == 'pdf' and href_attr:
                        paper['pdf_link'] = href_attr
                        break
                
                papers.append(paper)
        except Exception as e:
            logger.error(f"XML parsing error: {e}")
        
        return papers


class SerpApiClient:
    """SerpApi 客户端 - 搜索 Google Scholar 和 Semantic Scholar"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
    
    def search_google_scholar(self, query: str, max_results: int = 10) -> List[dict]:
        """搜索 Google Scholar"""
        papers = []
        
        try:
            params = {
                "engine": "google_scholar",
                "q": query,
                "api_key": self.api_key,
                "num": max_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if "organic_results" in data:
                for item in data["organic_results"]:
                    pub_info = item.get('publication_info', {})
                    paper = {
                        'id': item.get('result_id', ''),
                        'title': item.get('title', ''),
                        'summary': item.get('snippet', ''),
                        'authors': self._extract_authors(pub_info),
                        'link': item.get('link', ''),
                        'published': self._extract_year(pub_info),
                        'source': 'Google Scholar'
                    }
                    papers.append(paper)
            
            logger.info(f"Google Scholar found {len(papers)} papers")
        except Exception as e:
            logger.error(f"Google Scholar search error: {e}")
        
        return papers
    
    def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[dict]:
        """通过 SerpApi 搜索 Semantic Scholar"""
        papers = []
        
        try:
            params = {
                "engine": "google_scholar",
                "q": f"site:semanticscholar.org {query}",
                "api_key": self.api_key,
                "num": max_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if "organic_results" in data:
                for item in data["organic_results"]:
                    pub_info = item.get('publication_info', {})
                    paper = {
                        'id': item.get('result_id', ''),
                        'title': item.get('title', ''),
                        'summary': item.get('snippet', ''),
                        'authors': self._extract_authors(pub_info),
                        'link': item.get('link', ''),
                        'published': self._extract_year(pub_info),
                        'source': 'Semantic Scholar'
                    }
                    papers.append(paper)
            
            logger.info(f"Semantic Scholar found {len(papers)} papers")
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
        
        return papers
    
    def _extract_authors(self, publication_info) -> List[str]:
        """从出版信息中提取作者"""
        if not publication_info:
            return []
        
        if isinstance(publication_info, dict):
            authors = publication_info.get('authors', [])
            if isinstance(authors, list):
                return [a.get('name', '') for a in authors if isinstance(a, dict) and a.get('name')]
            summary = publication_info.get('summary', '')
            if summary:
                publication_info = summary
            else:
                return []
        
        if not isinstance(publication_info, str):
            return []
        
        authors_match = re.match(r'^([^-\\d]+)', publication_info)
        if authors_match:
            authors_str = authors_match.group(1).strip()
            return [a.strip() for a in authors_str.split(',') if a.strip()]
        return []
    
    def _extract_year(self, publication_info) -> str:
        """从出版信息中提取年份"""
        if not publication_info:
            return ''
        
        if isinstance(publication_info, dict):
            summary = publication_info.get('summary', '')
            if summary:
                publication_info = summary
            else:
                return ''
        
        if not isinstance(publication_info, str):
            return ''
        
        year_match = re.search(r'\b(19|20)\d{2}\b', publication_info)
        if year_match:
            return year_match.group(0)
        return ''


async def search_arxiv(
    query: str,
    max_results: int = 20
) -> str:
    """
    搜索 arXiv 文献
    
    Args:
        query: 搜索关键词
        max_results: 最大结果数（默认20）
    
    Returns:
        JSON 格式的文献列表
    """
    logger.info(f"Searching arXiv for: {query}")
    
    async with ArxivClient() as client:
        papers = await client.search(query, max_results)
    
    if not papers:
        return json.dumps({"error": f"未找到关于 '{query}' 的文献"}, ensure_ascii=False)
    
    result = {
        "query": query,
        "total": len(papers),
        "papers": papers,
        "source": "arXiv"
    }
    
    logger.info(f"Found {len(papers)} papers from arXiv")
    return json.dumps(result, ensure_ascii=False, indent=2)


async def search_serpapi(
    query: str,
    max_results: int = 10
) -> str:
    """
    通过 SerpApi 搜索 Google Scholar 和 Semantic Scholar
    
    Args:
        query: 搜索关键词
        max_results: 每个来源最大结果数（默认10）
    
    Returns:
        JSON 格式的文献列表
    """
    logger.info(f"Searching SerpApi for: {query}")
    
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "SERPAPI_API_KEY 未配置"}, ensure_ascii=False)
    
    client = SerpApiClient(api_key)
    
    google_scholar_papers = client.search_google_scholar(query, max_results)
    semantic_scholar_papers = client.search_semantic_scholar(query, max_results)
    
    all_papers = google_scholar_papers + semantic_scholar_papers
    
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_lower = paper.get('title', '').lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(paper)
    
    result = {
        "query": query,
        "total": len(unique_papers),
        "papers": unique_papers,
        "source": "SerpApi (Google Scholar + Semantic Scholar)"
    }
    
    logger.info(f"Found {len(unique_papers)} unique papers from SerpApi")
    return json.dumps(result, ensure_ascii=False, indent=2)


async def search_all_sources(
    query: str,
    max_results_per_source: int = 10
) -> str:
    """
    并行搜索所有数据源（arXiv + SerpApi）
    
    Args:
        query: 搜索关键词
        max_results_per_source: 每个来源最大结果数
    
    Returns:
        JSON 格式的合并文献列表
    """
    logger.info(f"Searching all sources for: {query}")
    
    arxiv_task = search_arxiv(query, max_results_per_source)
    serpapi_task = search_serpapi(query, max_results_per_source)
    
    arxiv_result, serpapi_result = await asyncio.gather(
        arxiv_task, serpapi_task, return_exceptions=True
    )
    
    all_papers = []
    
    if isinstance(arxiv_result, str):
        try:
            arxiv_data = json.loads(arxiv_result)
            if "papers" in arxiv_data:
                for paper in arxiv_data["papers"]:
                    paper['source'] = 'arXiv'
                    all_papers.append(paper)
        except:
            pass
    
    if isinstance(serpapi_result, str):
        try:
            serpapi_data = json.loads(serpapi_result)
            if "papers" in serpapi_data:
                all_papers.extend(serpapi_data["papers"])
        except:
            pass
    
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_lower = paper.get('title', '').lower().strip()
        if title_lower and title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_papers.append(paper)
    
    unique_papers.sort(key=lambda x: x.get('published', ''), reverse=True)
    
    result = {
        "query": query,
        "total": len(unique_papers),
        "papers": unique_papers,
        "sources": ["arXiv", "Google Scholar", "Semantic Scholar"]
    }
    
    logger.info(f"Total unique papers found: {len(unique_papers)}")
    return json.dumps(result, ensure_ascii=False, indent=2)


async def analyze_trends(papers_json: str) -> str:
    """
    分析文献趋势
    
    Args:
        papers_json: JSON 格式的文献列表
    
    Returns:
        趋势分析结果
    """
    try:
        data = json.loads(papers_json)
        papers = data.get("papers", [])
        
        if not papers:
            return json.dumps({"error": "无文献数据"}, ensure_ascii=False)
        
        years = defaultdict(int)
        categories = defaultdict(int)
        keywords = defaultdict(int)
        
        for paper in papers:
            year = paper.get('published', '')[:4]
            if year:
                years[year] += 1
            
            for cat in paper.get('categories', []):
                categories[cat] += 1
        
        trends = {
            "total_papers": len(papers),
            "year_distribution": dict(sorted(years.items(), reverse=True)),
            "top_categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]),
            "date_range": {
                "earliest": min(p.get('published', '9999') for p in papers),
                "latest": max(p.get('published', '0000') for p in papers)
            }
        }
        
        return json.dumps(trends, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Trend analysis error: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


async def generate_bibtex(papers_json: str) -> str:
    """
    生成 BibTeX 引用
    
    Args:
        papers_json: JSON 格式的文献列表
    
    Returns:
        BibTeX 格式的引用列表
    """
    try:
        data = json.loads(papers_json)
        papers = data.get("papers", [])
        
        if not papers:
            return "无文献数据"
        
        bibtex_list = []
        
        for i, paper in enumerate(papers, 1):
            authors = " and ".join(paper.get('authors', ['Unknown']))
            year = paper.get('published', '')[:4]
            title = paper.get('title', 'Untitled')
            arxiv_id = paper.get('id', '')
            
            key = f"arxiv{arxiv_id.replace('.', '')}"
            
            bibtex = f"""@article{{{key},
  title={{{title}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}},
  url={{https://arxiv.org/abs/{arxiv_id}}}
}}"""
            bibtex_list.append(bibtex)
        
        return "\n\n".join(bibtex_list)
    except Exception as e:
        logger.error(f"BibTeX generation error: {e}")
        return f"生成失败: {str(e)}"


async def save_review_to_file(
    content: str,
    filename: str = "literature_review.md"
) -> str:
    """
    保存文献综述到文件
    
    Args:
        content: 综述内容
        filename: 文件名
    
    Returns:
        保存结果
    """
    output_dir = "coding"
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Review saved to {filepath}")
    return f"文献综述已保存到: {filepath}"


def create_tools() -> List[FunctionTool]:
    """创建工具列表"""
    return [
        FunctionTool(search_all_sources, description="并行搜索所有数据源（arXiv + Google Scholar + Semantic Scholar）"),
        FunctionTool(search_arxiv, description="仅搜索 arXiv 文献"),
        FunctionTool(search_serpapi, description="通过 SerpApi 搜索 Google Scholar 和 Semantic Scholar"),
        FunctionTool(analyze_trends, description="分析文献趋势"),
        FunctionTool(generate_bibtex, description="生成 BibTeX 引用"),
        FunctionTool(save_review_to_file, description="保存文献综述到文件")
    ]


async def main():
    """主函数"""
    logger.info("Starting literature review agent")
    
    model_client = OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "glm-4-flash"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", ""),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        }
    )
    logger.info("Model client configured")
    
    tools = create_tools()
    
    search_agent = AssistantAgent(
        name="Search_Agent",
        model_client=model_client,
        tools=[tools[0], tools[1], tools[2]],
        system_message="""你是一个专业的文献搜索助手。
你的任务是：
1. 根据用户的研究主题，**必须首先**使用 search_all_sources 工具并行搜索所有数据源
2. search_all_sources 会同时搜索 arXiv、Google Scholar、Semantic Scholar
3. 返回合并后的搜索结果，包括文献标题、作者、摘要、发布日期、来源等信息
4. 如果 search_all_sources 失败，可以尝试单独使用 search_arxiv 或 search_serpapi
5. 不要询问用户，自动选择最佳搜索策略

重要：始终优先使用 search_all_sources 工具！

请用中文回复。"""
    )
    
    analysis_agent = AssistantAgent(
        name="Analysis_Agent",
        model_client=model_client,
        tools=[tools[3]],
        system_message="""你是一个专业的文献分析助手。
你的任务是：
1. 分析文献趋势（年份分布、热门领域等）
2. 提取关键信息（研究方法、主要结论、创新点）
3. 识别研究热点和空白
4. 分析不同来源的文献特点

请用中文回复。"""
    )
    
    report_agent = AssistantAgent(
        name="Report_Agent",
        model_client=model_client,
        tools=[tools[4], tools[5]],
        system_message="""你是一个专业的文献综述撰写助手。
你的任务是：
1. 基于搜索和分析结果，撰写结构化的文献综述
2. 生成 BibTeX 格式的引用
3. 保存综述报告到文件

综述报告应包含：
- 研究背景
- 文献概述（按来源分类）
- 研究趋势分析
- 主要发现
- 研究空白与未来方向
- 参考文献

请用中文撰写报告。完成后保存文件并回复 TERMINATE。"""
    )
    
    logger.info("Agents created")
    
    termination = TextMentionTermination("TERMINATE") | ExternalTermination()
    
    team = RoundRobinGroupChat(
        participants=[search_agent, analysis_agent, report_agent],
        termination_condition=termination,
        max_turns=10
    )
    
    current_date = datetime.now(timezone.utc).strftime("%Y年%m月%d日")
    
    task = f"""请对以下研究主题进行文献综述：transformer attention mechanism

注意：
1. 当前日期是{current_date}，请在报告中使用正确的日期
2. 使用中文撰写综述报告
3. 搜索至少15篇相关文献
4. 分析研究趋势和热点
5. 生成 BibTeX 引用
6. 将报告保存为 literature_review_transformer.md"""
    
    logger.info(f"Task: {task}")
    
    max_retries = 5
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting team execution (attempt {attempt + 1}/{max_retries})")
            stream = team.run_stream(task=task)
            await Console(stream)
            logger.info("Literature review agent completed")
            return
        except RateLimitError as e:
            logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retries reached. Exiting.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
