import os
import warnings
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, initialize_agent
from langchain_core._api.deprecation import LangChainDeprecationWarning
import requests
import json
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser
from typing import Union
import re
from pydantic import BaseModel
import time

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# 新增图片验证模型
class ImageResult(BaseModel):
    url: str
    valid: bool = False
    verified_at: float = None

# ------------------------
# 网页搜索（使用API）
# ------------------------
def web_scraper(keyword: str) -> str:
    """使用DuckDuckGo官方API"""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": keyword,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1
            },
            timeout=15
        )
        results = []
        for item in response.json().get("Results", [])[:3]:
            if url := item.get("FirstURL"):
                results.append(f"{item.get('Text', '')}: {url}")
        return "\n".join(results) if results else "无搜索结果"
    except Exception as e:
        return f"API错误：{str(e)}"

# ------------------------
# 图片搜索工具
# ------------------------
def validate_image(url: str, max_retries=2) -> bool:
    """验证图片链接有效性"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    for _ in range(max_retries):
        try:
            response = requests.head(
                url,
                headers=headers,
                timeout=10,
                allow_redirects=True
            )
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if any(ct in content_type for ct in ['image/jpeg', 'image/png']):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def image_scraper(query: str) -> str:
    """返回验证通过的图片链接"""
    try:
        # 使用实际API密钥
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "tbm": "isch",
                "api_key": os.getenv("SERPAPI_KEY", ""),
                "ijn": "0"
            },
            timeout=20
        )

        # 获取并验证图片链接
        valid_images = []
        for img in response.json().get("images_results", [])[:5]:  # 验证前5个
            url = img.get("original")
            if url and validate_image(url):
                valid_images.append(url)
            if len(valid_images) >= 2:  # 返回前2个有效链接
                break

        return "\n".join(valid_images) if valid_images else "未找到有效图片"
    except Exception as e:
        return f"图片服务错误：{str(e)}"

# ------------------------
# 本地检索，统一数据字段
# ------------------------
def local_search(keyword: str) -> str:
    try:
        with open("local_articles.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            matches = [
                          f"{item['title']}: {item['url']}"  # 统一使用英文字段
                          for item in data
                          if keyword.lower() in item.get("content", "").lower()
                      ][:3]
            return "\n".join(matches) if matches else "无本地数据"
    except Exception as e:
        return f"本地错误：{str(e)}"


# ------------------------
# 更新提示模板
# ------------------------
react_zh_template = """**请严格遵循格式要求！**

可用工具：{tool_names}

格式说明：
1. 图片链接已自动验证有效性
2. 必须包含至少1个图片结果
3. 最终答案需包含清晰分类

格式模板：
Question: 需要回答的问题
Thought: 分步中文思考
操作: 工具名称（必须精确匹配 {tool_names}）
操作输入: 工具输入
Observation: 斗兽场介绍: https://example.com/colosseum
...(重复次数小于3)
最终答案: 
文章推荐：
1. [罗马必游景点](https://example.com/top-sights)
图片资源：
1. ![斗兽场夜景](https://example.com/colosseum-night.jpg)
本地匹配：
1. [罗马交通指南](https://local.com/transport)

现在处理：

Question: {input}
Thought:"""

# ------------------------
# 统一工具名称映射
# ------------------------
class ChineseOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(f"[DEBUG] Model Output:\n{text}\n{'-' * 50}")

        # 增强正则表达式匹配
        final_pattern = r"最终答案\s*[:：]\s*(.*)"
        action_pattern = (
            r"操作\s*[:：]\s*(网页搜索|图片搜索|本地搜索)"
            r"\s*\n操作输入\s*[:：]\s*(.*?)(?=\n|$)"
        )

        if final_match := re.search(final_pattern, text, re.DOTALL):
            return AgentFinish(
                return_values={"output": final_match.group(1).strip()},
                log=text
            )

        if action_match := re.search(action_pattern, text, re.DOTALL):
            tool_mapping = {
                "网页搜索": "web_search",
                "图片搜索": "image_search",
                "本地搜索": "local_search"  # 修正工具名称
            }
            tool_name = action_match.group(1).strip()
            if tool_name not in tool_mapping:
                raise ValueError(f"未知工具：{tool_name}")

            return AgentAction(
                tool=tool_mapping[tool_name],
                tool_input=action_match.group(2).strip(),
                log=text
            )

        raise ValueError(f"解析失败！请检查格式：\n{text}")

# ------------------------
# 初始化智能体（带工具名称映射）
# ------------------------
llm = OllamaLLM(
    model="deepseek-r1:1.5b",
    temperature=0.1,  # 降低随机性
    verbose=True
)

tools = [
    Tool(name="网页搜索", func=web_scraper,
         description="搜索网络文章，返回文章和链接，输入格式：'关键词'"),
    Tool(
        name="图片搜索",
        func=image_scraper,
        description="获取已验证的有效图片链接，输入图片描述"
    ),
    Tool(name="本地搜索", func=local_search,
         description="优先使用本地数据库检索，输入格式：'关键词'")
]

prompt = PromptTemplate.from_template(react_zh_template).partial(
    tool_names="web_search, image_search, local_db"  # 显式列出工具名
)

# 初始化参数调整
agent = initialize_agent(
    tools=tools,
    llm=llm,
    prompt=prompt,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=lambda e: f"解析错误，请检查格式：{str(e)}",
    max_iterations=5,  # 增加迭代次数
    early_stopping_method="generate"
)

# ------------------------
# 运行测试
# ------------------------
if __name__ == "__main__":
    query = "罗马"
    # 创建示例数据
    if not os.path.exists("local_articles.json"):
        sample_data = [{
            "标题": "罗马历史",
            "内容": "从共和国到帝国的演变过程",
            "链接": "https://example.com/roman-history"
        }]
        with open("local_articles.json", "w") as f:
            json.dump(sample_data, f, ensure_ascii=False)

    # 执行查询
    try:
        response = agent.invoke({
            "input": f"找到并列出关于{query}的文章和图片以及链接，优先使用网页搜索和图片检索"
        })
        print("\n最终结果：")
        print(response["output"])
    except Exception as e:
        print(f"执行失败：{str(e)}")