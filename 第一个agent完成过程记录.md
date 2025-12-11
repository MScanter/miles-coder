# 第一个 Agent 完成过程记录

## 已完成

- ✅ Agent 核心循环 - 外层循环等待用户输入，内层循环处理工具调用
- ✅ 工具调用和执行 - read_file、write_file、list_files
- ✅ 美化输出 - 工具调用显示图标和参数
- ✅ Thinking 思考过程显示 - 使用 DeepSeek reasoner 的 reasoning_content
- ✅ 多工具调用支持 - 用 for 循环遍历所有 tool_calls
- ✅ 标准化 API 格式 - msg_to_dict 转换，兼容所有 OpenAI API 服务
- ✅ Pydantic 参数定义 - 用 BaseModel 定义工具参数类
- ✅ 自动生成 TOOLS_SCHEMA - make_tool 函数替代手写 schema
- ✅ 启动欢迎信息 - 显示模型名和使用提示

---

## 下一步

1. 接入 MCP - Model Context Protocol 标准工具协议
2. 添加 Bash 执行工具 - 让 Agent 能运行命令
3. 流式输出 - 让回复逐字显示，体验更好
4. 错误处理优化 - 文件不存在等情况的友好提示