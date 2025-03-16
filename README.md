# MultiModusAgent

![image](https://github.com/user-attachments/assets/2cbb50f5-f312-4bee-b441-48a5c46d0588)

核心功能
  多工具协同
    网页搜索：通过 DuckDuckGo API 获取实时资讯
    图片搜索：调用 SerpAPI 并自动验证图片有效性
    本地检索：基于本地 JSON 知识库的关键词匹配
  智能响应格式
    文章推荐：3 个带链接的标题
    图片资源：2 个验证过的图片链接
    本地匹配：本地知识库相关结果
  增强特性
    图片自动验证（支持 JPEG/PNG）
    多语言混合检索
    严格的格式校验机制
    工具调用监控与日志记录

.
├── src/
│   ├── agents/          # 智能体定义
│   ├── tools/           # 工具实现
│   ├── parsers/         # 响应解析器
│   └── utils/           # 通用工具函数
├── data/
│   └── local_articles.json  # 本地知识库
├── tests/               # 测试用例
├── logs/                # 运行日志
└── .env                # 环境变量配置

关键参数
  参数名	            描述	              推荐值
  temperature	      生成文本的创造性程度	0.1-0.3
  max_iterations	  最大工具调用次数	    5
  timeout	API       请求超时时间（秒）	  15-20
  valid_image_types	有效图片类型	        ["image/jpeg", "image/png"]
