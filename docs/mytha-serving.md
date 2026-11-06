# LoRA Writer 与 Mytha 联合运行

2026-10-27。协议以本项目 `src/prompt_builder.py` 和实际训练导出为基准。Mytha 负责故事规划、检查、调度、装配和状态管理；本服务只负责一次正文生成，不承担规划或审查。

## 可选部署边界

本项目是 Mytha 的可选外部 Writer 扩展服务，不是 Mytha 的安装依赖。Mytha 普通用户继续使用原有模型服务，无需本仓库、Python、CUDA、本地权重或独立显卡。

服务可部署在有算力的本机、局域网主机或云端；Mytha 客户端通过 HTTP 连接。仅部署模型的设备承担显存和计算开销。不要将训练环境和权重打包进 Mytha 安装包，也不要在 Mytha 启动时自动启动或下载本服务。

在 Mytha 项目的 Agent 设置中主动启用“可选扩展：LoRA Writer”才会用于新写整章。停用扩展并重新发起任务可恢复普通写作；停止/卸载本服务不影响未启用扩展的项目。已启用项目遇到服务不可用会提示检查服务或停用扩展，不静默改用其他模型。

## 启动

使用已有 `.venv`。服务不新增第三方依赖；本次验证环境为 torch 2.14.1、transformers 5.20.2、peft 0.21.2、accelerate 1.15.0。默认自动选择 CUDA，否则 CPU。只加载本地文件，不自动下载模型，不执行训练。

```bash
.venv/bin/python src/serve_writer.py \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --adapter output/tara_webnovel_7books \
  --context-tokens 4096 --output-tokens 1024
```

默认地址 `http://127.0.0.1:8765/v1`。权重只加载一次，推理串行执行；并发请求返回 429。适配器必须存在，不会偷偷退回基座模型。Ctrl+C 关闭服务。启动参数中的适配器只是现有权重示例，并不表明它已按最新协议重新训练。

可通过 `TARA_API_KEY` 环境变量配置认证；非回环地址必须配置该变量。Mytha 模型设置中填写相同密钥。局域网部署需填写实际服务主机地址，不能继续使用客户端的 127.0.0.1。

## 发现与配置

`GET /v1/models` 返回唯一模型、revision、协议、提示词源码 SHA-256、上下文和输出上限。模型 ID 包含基座/适配器文件、tokenizer 配置、提示模板、服务实现和预算参数的内容指纹。修改这些内容后重启服务会得到新 ID；旧 ID 的生成请求返回 404，不能静默切换模型。运行期间不要修改模型目录。

在 Mytha 仓库运行：

```bash
python3 scripts/tara/check-prompts.py /path/to/lora/src/prompt_builder.py
node scripts/tara/run-smoke-service.mjs http://127.0.0.1:8765/v1
```

第二条命令先校验服务协议与模板哈希，再通过 Mytha 的真实 OpenAIProvider 发起最多 32 tokens 的短生成；输出 modelSettings 和 taraWriter 配置，不写入个人设置或项目。使用认证时给此命令设置相同 `TARA_API_KEY`。

将输出中的 endpoint、model、maxTokens、maxContextTokens 配置到 Mytha 的模型设置，Provider 选 OpenAI 兼容；然后在项目 Agent 设置选择此 Writer，并将 revision 填入适配器版本。内部模型配置 ID 可以使用 Mytha 自动生成的值，Agent 的 modelId 应选择该配置。规划模型必须另行配置，不要将正文专用 LoRA 同时当作规划器或审查器。

模型升级后重新运行探测、更新模型名称和 revision。既有 Mytha 检查点会因模型配置/版本改变而失效。

## 请求与失败约定

- `POST /v1/chat/completions`，UTF-8 JSON，显式 Content-Length。
- 仅支持 `stream=true`、单条 system + 单条 user、单个输出。拒绝工具、非空 stop 序列及多候选请求。
- 输出为标准 SSE `delta.content`，结尾携带 `finish_reason=stop|length`，随后 `[DONE]`。错误以 HTTP JSON 或流内 error 返回，不能当作成功。
- tokenizer 使用与训练一致的聊天模板和 `add_special_tokens=False`。输入从不截断；实际提示 token 数加输出预算超限时在生成前返回 400。
- 输出预算取请求 max_tokens 与服务输出上限的较小值。上下文不足时不再暗中缩小输出预算；由调用端调整输入或 max_tokens 后重试。
- Mytha 把 `length` 视为未完成正文，保留检查点但禁止装配落盘。
- 客户端断开后通过心跳检测并设置停止条件；正在执行的 GPU 步骤结束后释放生成锁，期间不接受第二次生成。

## 验证与限制

```bash
.venv/bin/python -m unittest discover -s tests -v
```

新增服务测试覆盖认证、版本错误、上下文超限、SSE、截断、生成异常、取消和忙碌状态。Mytha 增加中文 UTF-8 跨分块、错误流及取消回归。

本次已使用现有 `tara_webnovel_7books` 权重完成 Mytha Provider → HTTP 服务 → GPU 生成的真实短调用。短测试主动限制输出，返回 length 是预期行为，只证明传输链路。当前协议哈希描述服务/提示版本，不证明现有权重的训练数据版本；尚未完成最新协议权重的整章质量验证，也未解决 Mytha S08 审查可靠性问题。
