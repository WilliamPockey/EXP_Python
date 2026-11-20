# Citation Grounding & Faithfulness Verifier

第一阶段交付一个可在本地启动的 `POST /verify` 服务。默认使用固定样例模拟模型调用，无需 API Key。题目中的 SQLite WAL 样例会返回四条声明，依次为 `CONTRADICTED`、`SUPPORTED`、`SUPPORTED`、`UNSUPPORTED`。[题目原文](https://gist.githubusercontent.com/wsxiaoys/849522ee389e0ce359ef444653084484/raw/435040b6643cce4c1124c4d64e68d60f298d7039/assignment.md)

## 已实现

- `app.py` 使用 FastAPI 提供 `POST /verify`，接收 `context_passages` 和 `generated_answer`，返回包含 `claim_text`、`verdict`、`reasoning` 和 `citations` 的 `claims` 列表。
- `verifier.py` 定义请求和响应类型、`VerificationModel` 接口以及确定性校验。`SUPPORTED` 和 `CONTRADICTED` 必须各自提供至少一条引用，`UNSUPPORTED` 的引用必须为空。`index` 必须是有效的零基整数索引。`quote` 必须是指定段落中的非空、连续原文，校验保留原始大小写和空白。
- `demo_model.py` 从 `examples/assignment.json` 读取请求和预设报告。它还支持原样例回答配空上下文，以及 `Please check the documentation.` 这条纯建议。空白回答由服务直接返回空声明列表。
- `tests/test_verify.py` 通过 HTTP 测试客户端验证样例、错误引用、请求和响应校验，以及上游异常处理。

若接入模型服务，实现 `VerificationModel.verify(request)`，让它返回解码后的 JSON 对象，再传给 `create_app(model)`。适配器负责为上游调用设置超时，并将模型请求失败和超时分别映射为 `ModelError` 和 `TimeoutError`。默认服务用预设样例演示声明拆分和语义判定，第三阶段评测通过独立 runner 调用本地 LLM。

## 启动服务

在 `Code` 仓库根目录运行以下 PowerShell 命令，推荐使用 Python 3.11 或更新版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

服务启动后，可在 [本地接口文档](http://127.0.0.1:8000/docs) 查看请求与响应结构。

## 验收

### 手工调用题目样例

保持服务运行，再在仓库根目录打开一个 PowerShell 窗口。

```powershell
$sample = Get-Content -Raw -Encoding UTF8 .\examples\assignment.json | ConvertFrom-Json
$requestBody = ConvertTo-Json -InputObject $sample.request -Depth 10
$response = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/verify `
  -ContentType 'application/json' -Body $requestBody
$response | ConvertTo-Json -Depth 10
```

预期返回 HTTP `200` 和四条声明。第一条引用段落 `0`，反驳通过 NFS 跨主机使用 WAL。第二、三条引用段落 `1`，分别支持辅助文件与自动 checkpoint 阈值。第四条关于 AES-256 加密，判定为 `UNSUPPORTED`，引用为空。

### 自动验收

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试无需启动 Uvicorn，也无需模型账户。第一阶段验收通过了全部 15 项测试，并用运行中的 Uvicorn 服务手工调用题目样例，确认返回 HTTP `200` 和预期的四条判定。

| 场景 | 预期行为 |
| --- | --- |
| 题目样例或使用了合法的原文片段 | `200`，返回校验后的报告 |
| 空白回答或指定的纯建议 | `200`，返回 `{"claims": []}` |
| 原样例回答配空上下文 | `200`，保留四条声明，全部标记为 `UNSUPPORTED`，引用为空 |
| 请求字段类型错误 | `422`，模型不被调用 |
| 引文伪造或改写、引用指向错误段落、索引越界，或判定与引用数量不符 | `502`，拒绝整份模型输出 |
| 模型响应结构不合法，或上游请求失败 | `502` |
| 默认模拟模型收到样例范围外的非空请求 | `503` |
| 模型抛出超时错误 | `504` |

## 第二阶段评测数据集

数据集包含 32 条自编英文技术样例和 51 条逐条标注的原子声明，涉及备份、队列、缓存、导出、迁移和传输配置。先定义 [标注规则 v1](data/annotation_rules.md)，再制作 [开发集](data/dev.jsonl) 和 [留出集](data/holdout.jsonl)。开发集用于迭代，留出集用于配置冻结后的评测。

### 数据格式

两个文件都是 UTF-8 JSONL，每行一个完整样例。待评测方法只接收 `request`，评测器读取其余字段。

| 字段 | 类型与含义 |
| --- | --- |
| `id` | 稳定的样例 ID，例如 `dev-09` 或 `holdout-17`。 |
| `category` | 样例的主要考点，取值见覆盖范围表。 |
| `request` | 与 `POST /verify` 请求一致，含 `context_passages` 字符串数组和 `generated_answer` 字符串。 |
| `expected_claims` | 按回答顺序标注的原子声明数组。每条含 `claim_text`、`verdict`、`reasoning` 和 `evidence_sets`。纯建议和空白回答用 `[]`。 |
| `expected_claims[].evidence_sets` | 证据组合数组。外层选任意一个充分组合，内层引用合起来提供各项前提。每条引用含零基 `index` 及原文子串 `quote`。`UNSUPPORTED` 使用 `[]`。 |

例如 `dev-09` 的声明可以用段落 `0` 和 `1` 一起支撑，也可以用段落 `0` 和 `2` 一起支撑。只引用段落 `0` 能确定启用的配置，却无法证明其压实周期。

### 覆盖范围

| `category` | 开发集 | 留出集 | 考点 |
| --- | ---: | ---: | --- |
| `fully_supported` | 3 | 4 | 同义改写、并列拆分、同义重复、条件与单位换算。 |
| `distortion` | 3 | 4 | 相邻数值、条件调换、否定反转、阈值边界与错用时间单位。 |
| `extrinsic` | 2 | 5 | 真实背景事实、合理但无证据的细节、未证实的因果和过度概括。 |
| `multi_passage` | 2 | 5 | 跨段支持和反驳、三段推导、替代证据、相似但范围不同的干扰段落。 |
| `boundary` | 2 | 2 | 纯建议、空白回答、空上下文、同一范围的冲突证据。 |
| 合计 | **12** | **20** | **32 条样例**。 |

声明级标签共 `SUPPORTED` 28 条、`CONTRADICTED` 12 条、`UNSUPPORTED` 11 条。两组均包含三种判定，部分样例同时包含不同判定的声明。

### 数据集验收

安装上面的依赖后，在 `Code` 仓库根目录执行以下命令。校验直接读取文件，无需启动服务或调用模型。

```powershell
.\.venv\Scripts\python.exe .\validate_dataset.py
```

预期输出开发集 12 例、19 条声明，留出集 20 例、32 条声明，最后一行为 `Validated 32 cases and 51 claims.`。校验器检查 JSONL 结构、样例 ID 和请求的唯一性、分集覆盖、全部证据组合的判定与引用约束，以及跨段类别的引用跨度。失败时命令以非零状态退出，记录级错误会给出文件名和行号。

语义验收按[标注规则](data/annotation_rules.md)对照回答和证据复核。可用 `dev-06` 检查调换条件后的两条反驳，`holdout-16` 检查三段证据是否都必要，`holdout-18` 检查不同传输配置的超时能否迁移，以及 `holdout-20` 检查同一范围的冲突如何判定。

## 第三阶段基线与评测

- 非 LLM 基线（`lexical`）把回答按句切分，用词项 Dice 相似度选最接近的来源句。支持要求相似度至少为 `0.72`，并检查数值和否定词。句子核心词序相同且出现精确数值或否定差异时，才尝试判为反驳。证据直接取自选中的原文句。
- 最简 LLM 基线（`llm`）把简短指令和请求 JSON 放在一条用户消息中，每例调用一次 Ollama `/api/chat`。输出使用现有报告类型生成的 JSON Schema。验收配置为温度 `0`、seed `0`、最多生成 2048 token，关闭思考输出。[Ollama Chat API](https://docs.ollama.com/api/chat) · [DeepSeek-R1 提示建议](https://github.com/deepseek-ai/DeepSeek-R1#usage-recommendations)
- `evaluate.py` 统一读取标注、校验响应和计算分数。两种方法都仅接收样例的 `request`。可选的 JSON 报告包含每例的预测、声明对齐、失败原因、延迟及 token 用量。

### 运行方法

在 `Code` 仓库根目录执行。只运行非 LLM 基线时无需 Ollama。

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --split dev --method lexical
```

运行两种基线时，启动本地 Ollama，确认 `ollama list` 中存在 `deepseek-r1:1.5b`。首次使用可执行 `ollama pull deepseek-r1:1.5b`。

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --split dev --method baselines `
  --model deepseek-r1:1.5b --output .\.runs\stage3-dev.json
```

可用 `--method llm` 单独运行 LLM，或用 `--ollama-url` 指定非默认端口。运行遇到单例超时、上游错误或无效报告时，会记录状态并继续。LLM 的所有请求均为超时或上游错误时，命令写出报告后以非零状态退出。

### 指标

| 指标 | 计算方式 |
| --- | --- |
| 声明抽取 Precision、Recall、F1 | 要求数值和否定标记一致，按词项 Dice 相似度从高到低一对一匹配，阈值为 `0.65`。分别以已校验的预测数和金标数作 Precision、Recall 的分母。 |
| 判定 Macro-F1 | 在成功对齐的声明上分别计算三类 F1，再取平均。缺席类别按 0 计。 |
| 证据覆盖率 | 分母是需要引文的全部金标声明。分子要求声明对齐、判定正确，且预测在相同段落中覆盖任一金标证据组合的全部原文。 |
| 原文引用通过率 | 对响应校验前可以读取的引用计算有效引用比例。缺失的引用另由证据覆盖率及响应校验反映。 |
| 端到端样例正确率 | 样例的报告有效、所有声明一对一匹配、每条判定正确且所需证据齐全，才计为通过。 |
| 用量和延迟 | 记录模型调用数、上游报告的输入与输出 token 及调用和校验的 p50、p95 延迟。延迟采用最近秩百分位，包含失败请求。 |

失败样例保留在总样例数中，保留金标声明并按零条预测计分。自动对齐基于词面，同义改写和并列句可能误配或漏配，可在逐例报告中按标注规则复核。证据分数采用保守的原文包含判定，进一步截短但语义充分的引用可在语义复核中识别。

### 开发集验收结果

使用 12 条开发样例、19 条金标声明，在本机 Ollama 0.23.3 及 NVIDIA RTX 3090 Ti 上实跑。LLM 使用已加载的 `deepseek-r1:1.5b`，本轮为本地推理，云端 API 费用为 0，设备开销未计量。

| 指标 | 非 LLM | 最简 LLM |
| --- | ---: | ---: |
| 通过响应校验 | 12/12 | 11/12 |
| 声明对齐数 / 预测数 / 金标数 | 10 / 14 / 19 | 0 / 0 / 19 |
| 声明抽取 Precision / Recall / F1 | 0.714 / 0.526 / 0.606 | 0 / 0 / 0 |
| 判定 Macro-F1（已对齐声明） | 0.167 | 0 |
| 证据覆盖 | 0/15 | 0/15 |
| 原文引用通过 | 1/1 | 0/1 |
| 端到端通过 | 1/12 | 1/12 |
| p50 / p95 延迟（ms） | 0.047 / 0.090 | 328.231 / 7590.859 |
| 模型调用数 | 0 | 12 |
| 上游输入 / 输出 token | — | 1962 / 313 |

- 非 LLM 在 `dev-06` 合并了两条条件调换后的错误声明，并因词面重叠判为 `SUPPORTED`。`dev-09` 需要组合两个段落，单句基线判为 `UNSUPPORTED`。
- LLM 的 11 份有效报告均返回空声明，只有纯建议样例 `dev-11` 端到端通过。`dev-07` 提交的引文在指定段落中找不到逐字匹配，整份报告被拒绝。表中分数对应这一本地模型和提示配置。
- 运行 `python -m unittest discover -s tests -v` 共通过 20 项测试。新增测试覆盖数字和否定冲突、LLM 请求仅含输入、声明一对一对齐、证据组合及无效报告计分。

## 第四阶段两步核验

1. **拆分声明。** 用有限的英文规则预拆显式并列谓词，只向模型提供编号后的回答片段。模型为每个片段返回独立事实，保留条件、单位和否定。程序检查片段 ID 并按顺序合并结果，去掉文本重复。空白和可直接识别的纯建议返回空列表，空上下文中的声明保留并统一判为 `UNSUPPORTED`。
2. **批量核验。** 一次传入全部原子声明及带零基编号的完整上下文，同时提供词面相关度靠前的段落编号。程序检查每个声明 ID 恰好有一项结果，再检查判定与引用数量、段落索引和原文子串。跨段判定可以同时引用多个段落。
3. **一次引用修复。** 只提交引用不合规的声明和对应错误，保留已通过的结果。修复模型从编号的来源句中选取证据，程序还原原始段落索引和逐字引文并再次执行校验。二次校验失败时，`POST /verify` 返回 `502`，评测将该例记录为 `invalid_output`。

### 开发集迭代

- 第三阶段 `dev-06` 把交换条件的两条事实合并，`dev-09` 的证据分布在不同段落，`dev-07` 的 LLM 引用无法逐字匹配来源。第一版因此将声明拆分和核验分成两次调用，并按 ID 只重试引用不合规的声明。
- 第一版中 `dev-01`、`dev-02` 仍出现整段回显，`dev-11` 把纯建议当作声明。7 次引用修复均得到格式有效的报告，这些待修复声明最终都被归为 `UNSUPPORTED`。
- 第二版把拆分输入改成逐句编号，加上纯建议的短路处理，并将修复改为选原文编号，以减轻模型重复抄写引文的负担。修复成功率衡量报告是否最终通过校验，判定质量和证据覆盖另按金标计分。
- 第三版针对 `dev-01` 和 `dev-12` 中残留的复合句增加并列谓词预拆，并给核验提供相关段落提示。`dev-07`、`dev-08` 在第二版修复时仍缺证据，第三版在修复输出 schema 中约束判定与证据数量。`dev-12` 最终拆出两条声明，在空上下文下均正确标为 `UNSUPPORTED`。`dev-05` 仍引用了不同缓存条件下的原文。

| 优化版开发集迭代 | 声明抽取 F1 | 判定 Macro-F1 | 证据覆盖 | 端到端通过 | 修复通过 / 尝试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一版 | 0.452 | 0.302 | 0/15 | 0/12 | 7/7 |
| 第二版 | 0.452 | 0.222 | 0/15 | 1/12 | 5/7 |
| 第三版（冻结） | 0.686 | 0.262 | 0/15 | 2/12 | 5/6 |

同一轮开发集对比中，非 LLM 和最简 LLM 的声明抽取 F1 分别为 `0.606` 和 `0.000`，端到端均通过 `1/12`。第三版用于留出集评测。

### 运行与验收

启动 Ollama 并安装模型后，可用统一 runner 重现实测。`all` 运行三种方法，`baselines` 运行两种基线，`optimized` 只运行优化版。两种 LLM 方法共用 `--model` 指定的模型、推理参数及评测口径。

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --split dev --method all `
  --model deepseek-r1:1.5b --timeout 40 --output .\.runs\stage4-dev.json
.\.venv\Scripts\python.exe .\evaluate.py --split holdout --method all `
  --model deepseek-r1:1.5b --timeout 40 --output .\.runs\stage4-holdout.json
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

要通过 HTTP 运行优化版，使用独立的 FastAPI 工厂。接口仍为 `POST /verify`，请求格式同第一阶段。可用 `GROUNDING_OLLAMA_URL` 指定 Ollama 地址，默认是 `http://127.0.0.1:11434`。

```powershell
$env:GROUNDING_MODEL = 'deepseek-r1:1.5b'
.\.venv\Scripts\python.exe -m uvicorn app:create_optimized_app --factory --host 127.0.0.1 --port 8000
```

报告记录实际调用数并累计拆分、核验和修复的 token 与延迟。原文引用通过率统计首次核验中模型直接给出的引文，修复后由程序还原的引文用于证据覆盖和端到端计分。

### 留出集对比结果

配置冻结后，在 20 条留出样例、32 条金标声明上各运行一次。最简 LLM 和优化版使用同一个本地 `deepseek-r1:1.5b`，环境为 Ollama 0.23.3 和 NVIDIA RTX 3090 Ti。两种方法的云端 API 费用均为 0，设备开销未计量。

| 指标 | 非 LLM | 最简 LLM | 两步优化版 |
| --- | ---: | ---: | ---: |
| 通过响应校验 | 20/20 | 15/20 | 19/20 |
| 声明对齐数 / 预测数 / 金标数 | 19 / 23 / 32 | 5 / 7 / 32 | 18 / 23 / 32 |
| 声明抽取 Precision / Recall / F1 | 0.826 / 0.594 / 0.691 | 0.714 / 0.156 / 0.256 | 0.783 / 0.563 / 0.655 |
| 判定 Macro-F1（已对齐声明） | 0.244 | 0.250 | 0.297 |
| 证据覆盖 | 3/25 | 3/25 | 2/25 |
| 原文引用通过 | 6/6 | 11/15 | 7/12 |
| 端到端通过 | 3/20 | 2/20 | 3/20 |
| 引用修复通过 / 尝试 | — | — | 13/14 |
| p50 / p95 延迟（ms） | 0.051 / 0.096 | 409.442 / 12085.298 | 7122.633 / 13442.760 |
| 实际模型调用数 | 0 | 20 | 52 |
| 上游输入 / 输出 token | — | 3429 / 2045 | 9444 / 4133 |

- 优化版的已对齐判定 Macro-F1 最高，端到端通过数与非 LLM 基线相同，证据覆盖为 `2/25`。`holdout-15` 判定正确，却只引用了租户的配置名称，缺少该配置禁止写入共享存储桶的段落。`holdout-20` 的修复报告正确将冲突证据判为 `UNSUPPORTED`。`holdout-09` 因理由为空返回 `invalid_output`。
- 全部 28 项自动测试通过，覆盖一次定向修复、跨段证据回填、保留已通过的声明、错误 ID 及修复后仍失败的 HTTP `502`。通过运行中的优化版 Uvicorn 服务调用 `dev-12`，实际收到 HTTP `200` 和两条无引用的 `UNSUPPORTED` 声明。
- 逐例报告保存在 `.runs/stage4-dev.json` 和 `.runs/stage4-holdout.json`，含预测、对齐、引用命中、修复状态、耗时和用量。
