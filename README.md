# ai-agent-learning

《深入理解 AI Agent》学习记录 —— 边读边跑实验，把书里的结论用自己的实测数据核对一遍。

## 📖 出处与致谢

本仓库是我的**个人学习笔记**，内容基于下面这本书：

| | |
| --- | --- |
| 书名 | **《深入理解 AI Agent》** |
| 作者 | **Bojie Li（李博杰）** |
| 原书仓库 | https://github.com/bojieli/ai-agent-book |
| 在线阅读 | https://bojieli.github.io/ai-agent-book/ |
| 开源许可 | [Apache License 2.0](https://github.com/bojieli/ai-agent-book/blob/main/LICENSE) |

**几点说明**：

- 笔记中的**书中原文**均以 `>` 引用块明确标出，**版权归原作者所有**；其余部分是我的批注、实测数据与思考。
- 本仓库是**非官方**的个人学习记录，与原作者无隶属关系，不代表原作者观点。
- 原书以 **Apache-2.0** 许可发布，本笔记对书中内容的引用遵循该许可的署名要求。
- 若原作者认为本仓库有不妥之处，请联系我，我会立即调整或删除。

> 如果这些笔记对你有帮助，**更建议直接去读原书**：[bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book) ⭐

## 目录

- [第 1 章 · 学习记录](chapter1学习笔记.md)
  - 核心概念：Agent = Model + Harness（harness 三层机制、工程范式五波演进）
  - 编排模式：工作流与自主
  - 实验 1-1 ★★ 上下文的关键作用（消融实验）
  - 实验 1-2 ★ Kimi K3 原生 Agent 能力
  - 实验 1-3 ★ GPT-5.6 原生 Deep Research 能力
  - 实验 1-4 ★ 文生图工作流与原生图像生成的对照

- [第 2 章 · 学习记录](chapter2学习笔记.md)
  - 上下文工程：上下文窗口的构成、Chat Template、KV Cache / Prompt Cache
  - 实验 2-1 ★ 本地 LLM 服务部署与工具调用
  - 实验 2-2 ★ 注意力机制可视化（含 thinking / answer 区域放大图）

- [第 3 章 · 学习记录](chapter3学习笔记.md)
  - 用户记忆与知识库：三层次评估、四种记忆格式、BM25、混合检索
  - 实验 3-6 ★★ 混合检索流水线（BM25 + Dense + RRF + 神经重排）

- [第 4 章 · 学习记录](chapter4学习笔记.md)
  - 工具：五类划分（感知 / 执行 / 协作 / 用户沟通 / 事件触发）、工具设计的通用原则
  - 工具生态：MCP 与 Skill Hub、层次化组织与主动工具发现
  - 实验 4-4 ★★ 执行工具 MCP 服务器

## 说明

- 书中原文用 `>` 引用块标出，其余为个人批注、实测数据与思考。
- 图片放在各章同级的 `*.assets/` 目录下。
- 实测产物路径（如 `chapter1/context/validation/...`）相对于原书仓库 `bojieli/ai-agent-book`。
