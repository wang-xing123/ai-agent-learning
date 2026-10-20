# 工具

工具分类，设计规则，工具管理，agent和工具的配合（感知执行协作事件触发、用户沟通）

## 工具的分类

感知，执行，协作，用户沟通，事件触发

## 工具设计的通用原则

### 能力的表达形式：专用工具、通用执行器与 Skill

skill的目录条目比工具schema便宜

通用工具，在沙箱中让agent自己配环境，自己用python实现各种功能

agent控制编程并执行这个工具，在代码中控制其他工具的执行，执行过程看不到，只会看到结果。code is everything

### 工具描述的艺术

什么时候触发

工具能力边界描述，说清楚不能做什么

如何描述工具和如何描述skill是通用的

### 参数传递的保真性

**静默参数注入**：

## 工具生态：MCP 与 Skill Hub



mcp提供工具，只读的资源，提示词模板

## 工具太多怎么办：层次化组织与主动工具发现

### 层次化组织与按需加载

**按需加载：只暴露索引。** 

按照信息源的类型把工具分成四个大类（主动搜索，读取文档，读取非结构化数据，读取结构化数据）

这个按语义相似度把和语义相关的tool注入上下文（利用RAG）

### 模型原生的主动工具发现

按需加载工具，在agent运行过程中如果发现需要使用某工具，就申请加载该工具到上下文。

或者三个通用性比较强的工具，网络搜索(扩大信息源)，code interpreter（用代码编写各种工具）,tool search tool（查询工具）



| 图中名称        | 真实对应                                                     | 说明                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **GitHub**      | [github/github-mcp-server](https://github.com/github/github-mcp-server) | 官方维护，有远程托管版（`api.githubcopilot.com/mcp/`）和本地版，是真实存在的 |
| **Weather**     | MCP 官方 Quickstart 示例服务器                               | 只有 `get_alerts` 和 `get_forecast` 两个工具，使用美国国家气象局（NWS）API；属于教程级示例，并非产品 |
| **Finance**     | 没有官方对应                                                 | 只有第三方实现（如基于 yfinance 的），这里的名称属于纯占位   |
| **ArXiv**       | 社区实现（如 `arxiv-mcp-server`）                            | 并非官方，工具名因实现而异（`search_papers`、`download_paper`、`read_paper`…） |
| **File System** | `@modelcontextprotocol/server-filesystem`                    | 官方参考实现，有 `read_file`、`write_file`、`list_directory`、`search_files`、`list_allowed_directories` 等 |

动态加载工具不能破坏静态前缀，只能加载到上下文的末尾，然后再agent的state状态栏中维护一份简短的tool工具列表

缺点，工具定义散在context的各个角落，

> **实验 4-1 ★★★：主动工具发现**
>
> 两组实验对比，1.一次性注入120个工具，2.仅仅保留元工具 web search,code_interpreter,discover_tools
>
> 需求->discover_tools->RAG嵌入向量匹配->加入匹配的工具define到context->agent state 中工具列表更新->

### Skills：把工具发现变成“按需查阅”

skill和tool的结合，在skill中写SOP标准操作流程，在流程中说明工具的使用说明

这里将不同工具打包到不同skill中，skill按需动态加载，在加载的过程中也把tool的内容加载进去。就不用借助RAG

## 感知工具



> **实验 4-2 ★★：感知工具 MCP 服务器**
>
> 自己构建mcp服务器，里面包含各种工具

### 多模态感知

#### 原生多模态处理

deepseek-v4.1-flash1017有原生多模态的能力

#### 提取为文本

#### 工具化多模态分析

> **实验 4-3 ★★：多模态信息提取：三种技术范式的对比分析**

## 执行工具

> 审批失败后不应简单重试，而应**将拒绝理由作为工具调用结果加入 Agent 的轨迹**。从提议模型的视角看，审批拒绝就像一次工具调用失败，返回了错误信息和修正建议——Agent 已经具备处理工具失败的能力，审批机制只是一个新的输入源。

> **实验 4-4 ★★：执行工具 MCP 服务器**

## 协作工具

1.通过task工具将任务交给子agent完成

2.human-in-the-loop人工介入

> **实验 4-5 ★★：协作工具 MCP 服务器**

## 本章小结

能力靠两条渠道分发：MCP 协议统一专用工具的接入，Skill Hub 用包管理器分发 `SKILL.md`

## 思考题
