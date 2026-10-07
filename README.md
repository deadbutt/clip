# 蝶殇工作台 (DieShang Workbench)

基于 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 二次开发的**本地视频字幕工作台**：

```
下载 → 转录 → 人声分离 → 说话人分离 → 字幕后处理 → 校对原稿 → 翻译 → 校对译文 → 编辑 → 烧录 / 切片
```

转录、字幕编辑和视频处理在本机执行。选择本地 HY-MT2 / OPUS-MT 时，翻译也在本机执行；使用 AI 翻译、AI 校对或 AI 切片分析时，会把相关字幕文本发送到你配置的服务。视频下载和首次模型下载需要联网。

---

## 功能总览

| 环节 | 实现 | 说明 |
| --- | --- | --- |
| 视频/音频下载 | yt-dlp（`tools/yt-dlp/yt-dlp.exe`） | 默认 `best[height<=1080]` 保留视频供烧录；输出 MKV 容器；默认 `--cookies-from-browser firefox`；实时进度（百分比/速度/ETA） |
| 语音转录 | faster-whisper（large-v3-turbo / medium） | 词级时间戳；VAD 用 faster-whisper 原生默认；`hallucination_silence_threshold=2.0` 抑制静音幻觉；支持热词 |
| 人声分离 | demucs（htdemucs） | 与 whisper 并行执行；`has_background_audio` 门控（词间隙/语音响度比 > 0.12 才启用人声轨）；whisper 始终吃原始音频 |
| 说话人分离 | pyannote speaker-diarization-3.1 | auto 模式自由聚类（上限 4 人）+ 时长占比 <10% 杂簇收编；跨说话人段用词级时间戳归属；说话人数量可指定 1-10 |
| 字幕后处理 | 词级重组 | 从词级时间戳重新断句，逗号软切（优先标点边界，避免在动词/介词/固定短语中间硬切）；合并 ≤4 权重碎段、拆分 >11s 长段；剔除重复幻觉 |
| 校对原稿 | 两遍架构 | Pass1 滑动窗口提出局部修正；Pass2 全片术语分析；查看建议后应用到源稿 |
| 翻译 | OpenAI 兼容 API / Ollama / 本地 HY-MT2 / 本地 Opus-MT | 多 profile 随时切换；本地引擎零 token 成本；校对跑在翻译前的源稿上；拆分/合并后自动标记需重译段落 |
| 校对译文 | LLM 源文-译文对照 | 独立检查任意翻译引擎的译文，标记漏译/错译/术语不一致；支持勾选后一键应用建议 |
| 字幕编辑器 | Web 内置 | 视频预览实时叠加字幕；Ctrl+Enter 光标拆分（词边界对齐）；⇊ 相邻合并；校对 diff 确认弹窗；SRT/ASS 外部编辑自动检测同步 |
| 烧录/切片 | ffmpeg（`tools/ffmpeg/ffmpeg.exe`） | ASS 字幕整片/片段烧录；音频统一重编码 AAC 192k（避免 Opus 兼容性问题） |

---

## 系统要求

- Windows（启动脚本为 `.bat`；核心代码跨平台，但未在其他系统测试）
- Python 3.10+（推荐 3.12）
- [uv](https://docs.astral.sh/uv/) 包管理器
- NVIDIA GPU 可选（转录/分离 CPU 可跑但慢）

## 安装

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[torch-runtime,diarization]"
```

可选依赖组：

| 组 | 内容 |
| --- | --- |
| `torch-runtime` | torch / torchaudio / librosa 等运行依赖；demucs 包需另行准备 |
| `diarization` | pyannote.audio（说话人分离必需） |
| `flash-attn` | 可选实验依赖；安装它不会自动加速 faster-whisper/CTranslate2，当前默认流程不需要 |
| `dev` | pytest / Ruff |

### 本地模型

首次使用按需运行根目录脚本下载模型到 `models/`（已 gitignore）：

| 脚本 | 模型 | 用途 |
| --- | --- | --- |
| `download_faster_whisper_large_v3_turbo.bat` | faster-whisper-large-v3-turbo | 转录主力模型（默认） |
| `download_faster_whisper_medium.bat` | faster-whisper-medium | 轻量备选 |
| `download_qwen3_asr.bat` | qwen3-asr-1.7b | 备选 ASR 后端（质量弱于 whisper，未进生产） |

pyannote 需要 HF Token（gated 模型），或提前本地化到 `models/pyannote-speaker-diarization-local`。
Demucs 需要额外安装兼容的 demucs 包，并准备 `models/demucs-htdemucs/955717e8.safetensors`；当前 `torch-runtime` 依赖组不包含 demucs，缺少组件时不会启用人声分离。OPUS-MT 需要提前准备 CTranslate2 模型目录和 SentencePiece tokenizer，当前代码不自动下载它。HY-MT2 也需要手动准备 GGUF 和 llama.cpp，见翻译一节。

### 分发给别人：推荐的最省事方案

当前最容易可靠交付的是**代码与工具包 + 独立模型包 + 目标电脑首次安装依赖**。项目依赖 Python、CUDA/torch、ffmpeg、llama.cpp 和多个大模型，发布内容需要同时管理这些版本。单个 exe 并不能省掉模型、外部工具和驱动兼容问题。

#### 发布目录（当前可用的源码交付方式）

从已验证的项目版本创建单独的发布目录，推荐结构如下：

```text
DieShang-Workbench/
├── moss_transcribe_diarize/
├── start.bat
├── pyproject.toml / uv.lock
├── models/                        # 只放准备离线使用的模型
├── tools/ffmpeg/                  # ffmpeg.exe + ffprobe.exe
├── tools/yt-dlp/                  # 可选：yt-dlp.exe
├── tools/llama-server/bin/        # llama-server.exe 及其 DLL
└── config/
    ├── hotwords.json
    └── protected_terms.json
```

程序和工具可以打成一个 zip，模型另打一个包并解压到 `models/`，方便以后只更新程序。**不要复制开发机的 `.venv` 当成便携环境**：它可能引用创建环境时的 Python 绝对路径，editable 安装也可能引用源码绝对路径。对方解压后需要按下一节建立自己的环境。

不要把包含真实 API Key 的 `config/llm_profiles.json` 放进公开压缩包；交付前删除它，或者只保留不含密钥的示例文件。`runs/`、`.pytest_cache/`、`.ruff_cache/`、`.git/` 和测试截图也不需要分发。

#### 收到压缩包后的首次安装

用户安装 uv 和合适的显卡驱动，在解压目录打开 PowerShell。下面是手动安装方式；有网络时 uv 可按需下载 Python：

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[torch-runtime,diarization]"
```

再把模型放入 `models/`，把 `tools/llama-server/` 和 `tools/ffmpeg/` 准备好，双击 `start.bat`。这些命令是开发安装入口，**并不保证复现开发机的 CUDA wheel 和全部版本**；正式发布应固定经过验收的依赖及 torch/torchaudio 下载源。也可先验证随包的 `uv.lock`，再使用 `uv sync --locked --extra torch-runtime --extra diarization`，不能把未验证的锁文件当成当前环境快照。

当前 `start.bat` 使用 NVIDIA CUDA / float16，默认自动检测语言。已知日语素材可明确指定 `--language ja`；CPU 用户应按启动一节手动指定设备。HY 的 llama.cpp 后端独立于 Whisper，`--device cpu` 不会自动把 HY 改成 CPU/Vulkan。

#### 未来的“解压即用”整合包

如果目标是让非技术用户不用安装 Python，建议另做 **Windows NVIDIA 版文件夹整合包**：携带可重定位的独立 Python runtime、固定依赖和工具，再由专用启动器使用相对路径启动。它需要新的打包/安装脚本，并在没有开发环境的新电脑上验证；当前仓库还没有实现这种发行包。随后可用安装器封装这个目录，而不是一开始把所有内容塞进一个巨大 exe。

建议分开发布程序包、模型包；更新时保留用户的 `config/` 和 `runs/`。同一批模型无需每次重新下载，默认词表只在首次安装时初始化，升级不要覆盖用户修改。

#### 分发前检查清单

1. 在一台干净的 Windows 电脑上解压测试，不要依赖开发机的 PATH、全局 Python 或缓存模型。
2. 双击 `start.bat`，确认页面能打开；上传一个短音频，确认转录、导出和翻译都能完成。
3. 如果要使用说话人分离，确认 `models/pyannote-speaker-diarization-local` 已完整携带，或给用户说明 HF Token 的配置方法。
4. 如果要使用 HY-MT2，确认 `models/Hy-MT2-1.8B-Q4_K_M.gguf`、`tools/llama-server/bin/llama-server.exe` 和同目录 CUDA DLL 一起携带。
5. 确认 NVIDIA 驱动分别满足 torch/CTranslate2 与 llama.cpp 的运行库要求。Vulkan 构建仅影响 llama.cpp/HY，不能替代 Whisper/torch 的 CUDA 后端。记录并交付工具/模型许可证，按各自条款确认权重和二进制的再分发方式；项目的 Apache-2.0 不自动覆盖所有依赖。
6. 把 `config/hotwords.json` 和 `config/protected_terms.json` 作为领域默认词表随包提供；用户之后可在页面修改。两者分别服务 Whisper 转录和翻译。

当前有部分 ASR 下载脚本，但没有覆盖所有模型的一键安装器。近期推荐先把“首次安装 + 环境检测”脚本做完整，固定一个 Windows NVIDIA 配置进行验收；需要免安装体验时再制作独立 runtime 整合包。

### 外部工具

- **ffmpeg / ffprobe**：优先使用 `tools/` 下的便携版（依次查找 `tools/ffmpeg/bin`、`tools/ffmpeg`，以及 `tools/` 下任何同时包含 `ffmpeg.exe` 与 `ffprobe.exe` 的子目录）；没有便携版时回退到系统 PATH。推荐便携版，避免和系统全局环境互相干扰
- **yt-dlp**：优先使用 `tools/yt-dlp/yt-dlp.exe`（依次尝试当前运行目录、项目根目录、启动文件所在目录）；都没有时回退到系统 PATH 中的 `yt-dlp`

---

## 启动

### Web 界面（推荐）

双击 `start.bat`，或：

```powershell
.venv\Scripts\mtd-subtitle-web.exe                # 默认 http://127.0.0.1:7860
.venv\Scripts\mtd-subtitle-web.exe --port 8080    # 自定义端口
```

`start.bat` 已自动附加常用参数（本地 `large-v3-turbo`、CUDA + float16、自动检测语言、HF 镜像、检测到本地 OPUS-MT 时启用本地翻译）。它适用于 NVIDIA 配置。手动启动时按需传参，完整列表见 `.venv\Scripts\mtd-subtitle-web.exe --help`，常用参数如下：

自动检测语言时不要传 `--language`，例如：

```powershell
.venv\Scripts\mtd-subtitle-web.exe --model models/faster-whisper-large-v3-turbo --device cuda --dtype float16
```

关闭浏览器标签不会停止后台服务。在启动窗口按 `Ctrl+C` 停止服务，再运行 `start.bat` 即可重启。修改 Python 后端或启动参数后需要重启；仅修改前端文件时刷新页面即可。网页保存词表后会用于后续任务，不需要重启，也不会自动改写已有任务。

> 本地 HY-MT2 不依赖 `--translator-provider`，只要 `tools/llama-server/` 和模型文件就位就会自动出现在翻译弹窗里。

**转录**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | `small` | Whisper 模型：HF 名称（如 `large-v3-turbo`）或本地模型目录 |
| `--device` / `--dtype` | `auto` / `auto` | 计算设备与精度；N 卡一般 `cuda` + `float16`，纯 CPU 用 `cpu` + `int8` |
| `--language` | 自动检测 | 固定语言代码（`en` / `zh` / `ja`…），设置后跳过检测 |
| `--beam-size` | `5` | 束宽；长视频 3 是速度/质量折中，1 只适合草稿 |
| `--decoding` / `--temperature` | `greedy` / `1.0` | 历史兼容入口；当前 Whisper runner 不使用它们控制解码 |
| `--prompt` | 内置 | Whisper 初始提示词（可放领域词引导） |

明确指定日语时可这样启动；默认不传 `--language`，由模型自动检测：

```powershell
.venv\Scripts\mtd-subtitle-web.exe --model models/faster-whisper-large-v3-turbo --device cuda --dtype float16 --language ja
```

自动检测不会注入英文示例；faster-whisper 会对原音频多个位置的短片段综合判断语言，避免片头或伴奏影响首段检测。选出的主要语言会用于整段转录和后续缺口补录，避免短片段重新检测后切换语言。任务的 `transcription_info` 保存实际语言、检测概率和取样结果，供复查使用。混合语言素材仍按主要语言处理；明确传入 `--language` 可覆盖检测。
| `--max-new-tokens` / `--max-len` | `8192` / `131072` | 历史兼容入口；当前 Whisper runner 忽略它们，不能通过调大来提升质量 |

**说话人分离**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--diarization-backend` | `none` | `auto`（自由聚类+杂簇收编）/ `pyannote` / `cluster` / `none` |
| `--pyannote-model` | `pyannote/speaker-diarization-3.1` | 可指向 `models/` 下的本地化目录 |
| `--speaker-count` | 自动 | 指定目标人数；先放宽上限自由聚类，再按嵌入质心收编到目标人数 |
| `--hf-token` | 读 `HF_TOKEN` | pyannote gated 模型下载凭据 |
| `--diarization-device` | `auto` | 说话人模型计算设备 |

**服务**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--host` / `--port` | `127.0.0.1` / `7860` | 监听地址与端口 |
| `--runs-dir` | `runs` | 任务产物根目录 |

**翻译**（`--translator-provider`：`openai` / `ollama` / `opus-mt`，默认 `openai`）：Web 端在首页"AI 服务"面板配置多份 profile 随时切换；CLI 直跑时用 `--translator-base-url` / `--translator-model` / `--translator-api-key` / `--translator-timeout`。本地 OPUS-MT 传 `--translator-provider opus-mt --translator-model <CT2模型目录> --translator-tokenizer-dir <tokenizer目录>`（默认 `models/opus-mt-en-zh`），`--translator-device` / `--translator-compute-type` 默认 `auto`。

Web 端翻译弹窗提供三个引擎，默认选 **本地 HY-MT2**：

| 引擎 | 说明 |
| --- | --- |
| 本地 HY-MT2 · 推荐 | 腾讯混元 HY-MT2 1.8B（Q4_K_M），经 llama.cpp 本地推理，支持保护词和短上下文；需要准备运行库与模型。 |
| 本地 Opus-MT · 极速 | Helsinki-NLP opus-mt-en-zh（CTranslate2）；适合快速初稿，不使用保护词或前后文。 |
| AI 服务 | 使用当前启用的 profile；质量、上下文上限和费用取决于模型及服务商。 |

上传与链接页面的 **Whisper 热词** 共用 `config/hotwords.json`。默认显示已保存词表且不可编辑，点击“修改”后可以添加、删除或清空，点击“保存”生效，也可“取消”。保存后的词表供后续转录使用；多个词用逗号或换行分隔。

翻译弹窗中的 **Protected terms** 使用相同的编辑方式，首次保存时写入 `config/protected_terms.json`，刷新页面后仍保留。HY-MT2 使用官方术语干预提示（例如 `Neuro translates to Neuro`），AI 服务使用“不翻译或改写这些词”的提示。两者均属于模型指令，不能保证每次完全遵循；OPUS-MT 不使用保护词。翻译使用已保存的词表，未保存的编辑不参与翻译。热词和保护词独立维护。

HY-MT2 默认参考前后各 **2 段原文字幕**，只翻译当前段。参考内容合计最多 600 字符，不跨越距当前段超过 12 秒的间隔，也不使用先前生成的译文。出现明显背景复述、提示标签或异常长输出时，该句自动退回无上下文重译；重译仍异常则保留原文。此检测不能保证发现所有语义错误，重要内容仍需校对。

可通过 `--translator-hy-context-window 0` 恢复不带上下文的方式；支持 `0/1/2/3`，数字表示每侧最多参考的字幕段数。此配置需要重启 Web 服务生效。实际比较方法及结果见 `docs/hy-context-evaluation-2026-09-20.md`。

### 翻译请求与上下文

| 项目 | HY-MT2 | AI 服务 |
| --- | --- | --- |
| 每个请求 | 当前 1 段原文 + 前后各最多 2 段参考 + 保护词 | 默认约 32 段待翻译字幕 + 前后各 2 段参考 + 保护词 |
| 调度 | 16 个请求线程，8 个 llama-server 槽位 | 按批依次请求；Ollama 默认缩小到 6 段 |
| 长句处理 | 保持原字幕边界，单段输出 | 同一说话人的连续未完句可组合理解，仍逐段返回 |
| 上下文容量 | 当前配置总 16384 tokens，已测构建每槽 2048；单条最大输出 512 tokens | 由服务端决定；程序不会自动用满模型上下文 |
| 历史记忆 | 不携带之前的译文 | 不携带完整对话历史，主要依靠批次与邻段原文 |

AI 组合单元最多 4 段、260 字符，同一说话人且间隔不超过 1.2 秒；返回格式或数量不符时会拆小重试。HY 的 `batch_size=2048` 指底层 token 计算批量，不是一次发送 2048 段字幕。

本机 482 段素材测试：HY 无上下文约 30～33 秒，前后各 2 段约 44 秒，仅为翻译阶段耗时，不含转录和烧录。模型能容纳的最大上下文不等于本项目实际发送的文本量。保护词和上下文都有局限，仍可能出现误译或参考内容混入。

热词用于 Whisper 主转录及整片识别不足时重试，当前局部缺口补录未传热词。AI 校对不会直接读取这两份词表；应用校对中的术语修正后，正确词会写回热词表，供以后转录使用。

**启用本地 HY-MT2**（两个文件都要就位，缺任一个前端会禁用"开始翻译"并给出提示）：

1. 从 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载 Windows 版并解压到 `tools/llama-server/`，确保 `llama-server.exe` 与 CUDA 运行库（`cudart64_*.dll` / `cublas64_*.dll` / `cublasLt64_*.dll`）在 `tools/llama-server/bin/`。CUDA 运行库在同页的 `cudart-llama-bin-win-cuda-*.zip` 里。GPU 驱动过旧时改用 `llama-*-bin-win-vulkan-x64.zip`（无需额外运行库）。
2. 把 HY-MT2 的 GGUF 放到 `models/Hy-MT2-1.8B-Q4_K_M.gguf`。

服务**按需启动、启动后保留**：首次点"开始翻译"时自动拉起 `llama-server`（本机曾测冷启动约 1.7 秒），翻译完成后继续运行，直到应用进程正常退出时清理。关闭网页不会停止它。默认端口 `8090`，可用 `--translator-hy-port` 改；当前配置显存约 2.4GB，后续转录时需考虑这部分占用。

相关参数：`--translator-hy-model`（GGUF 路径）、`--translator-hy-server-dir` / `--translator-hy-server-exe`（llama-server 位置，默认自动在 `tools/` 下查找）。

### 环境变量

`start.bat` 已预设好模型下载镜像与超时，手动跑 CLI 时按需自设：

| 变量 | 作用 |
| --- | --- |
| `HF_ENDPOINT` | HuggingFace 下载镜像；`start.bat` 默认 `https://hf-mirror.com`（国内加速） |
| `HF_TOKEN` | pyannote gated 模型下载凭据（等价于 `--hf-token`） |
| `HF_HUB_OFFLINE` | 设为 `1` 强制离线，全部使用本地模型 |
| `HF_HUB_ETAG_TIMEOUT` / `HF_HUB_DOWNLOAD_TIMEOUT` | 下载超时；`start.bat` 调大到 300s / 1800s |
| `HF_HUB_DISABLE_XET` | `start.bat` 设 `1`，规避 Xet 传输在部分网络下卡住的问题 |
| `MTD_SHARED_SPEAKER_EMBEDDING` | 默认启用同一窗口的重复声纹计算复用；设 `0` 使用原路径。仅对已验证的 pyannote.audio 4.0.7 / CUDA / WeSpeakerResNet34 推理启用，其他模型和版本保持原路径 |

声纹复用保留全部音频、窗口、说话人掩码与原聚类规则；三份完整素材的轮次与最终字幕回归一致，但 CUDA 批次变化仍可能产生微小浮点差异。实测与适用范围见 [性能报告](docs/performance-audit-2026-09-17.md)。回退时在启动应用前设置环境变量，例如 PowerShell：`$env:MTD_SHARED_SPEAKER_EMBEDDING = "0"`。

### 命令行

```powershell
mtd-subtitle --help
```

核心参数与 Web 版一致，另支持 `--segments-input`（跳过转录直接吃已有 segments JSON）、`--no-speaker-labeling`、`--out-dir` 等。

---

## Web 界面指南

第一次完整跑通一个任务的流程：

1. **建任务**：首页上传本地音视频（多文件自动排队），或贴 URL 由 yt-dlp 下载（进度实时显示）
2. **等待处理**：转录与 demucs 人声分离并行执行，说话人分离吃人声轨；完成后任务进入待审状态
3. **校对原稿**：编辑器里手动改，或跑 LLM 两遍校对，查看并应用修正建议
4. **翻译**（可选）：整片翻译；之后拆分/合并过的段落会标记"需重译"
5. **校对译文**（可选）：对照源稿检查译文，勾选建议后一键应用
6. **样式与命名**：设置页调字体/颜色/遮罩，给说话人起名
7. **产出**：导出 SRT/ASS，整片烧录或选段切片；所有产物在 `runs/<job_id>/`，重启服务不丢

### 首页（上传）

- **文件上传**：本地音视频，多文件排队
- **URL 任务**：在线视频直链，先下载后转录，进度实时显示
- **AI 服务**（折叠面板）：配置 LLM API，可存多份（DeepSeek / Ollama / 任意 OpenAI 兼容），"启用"按钮切换当前使用的配置；"测试连通"一键验证
- **下载 Cookie**：默认 Firefox（Chrome/Edge 的 App-Bound Encryption 导致 yt-dlp 读不到）

### 关注更新（YouTube / B 站）

点击侧栏「关注更新」，添加 YouTube 频道主页、@账号或 UC 频道 ID，或 B 站 UP 主主页 / 数字 UID。默认每 10 分钟检查一次，支持暂停、修改间隔和手动检查。

- 默认「提醒，确认后下载」：发现新投稿后显示未读提示，在更新列表点击「确认下载」加入任务队列。也可以选择「自动下载」。
- 默认只下载原视频；勾选「下载完成后自动转录」才会接着转录。只下载的任务完成后可查看文件位置、另存原视频或手动开始转录，文件保存在 `runs/<job_id>/`。
- 首次成功检查只记录近期投稿，不自动下载历史视频；切换到「全部近期视频」可以手动下载历史投稿。已记录的视频会去重，重复检查、重复确认和服务重启不会重复创建下载任务。
- 服务运行时持续轮询，关闭网页不影响检查和自动下载；关闭程序后暂停。可选的浏览器桌面提醒需要开启通知权限并保持页面打开，未读提示会保留到下次打开。
- 关注列表与更新记录保存在 `config/subscriptions.json`。首次成功关注时会保存最近 5 条投稿（标题、封面、发布时间；B 站还会显示平台返回的时长），服务每次启动和后台轮询都会再检查更新。长时间停机仍可能遗漏超出平台近期列表的投稿；此功能检查视频投稿，不监控动态、直播开播或提供即时推送。
- 付费、大会员、充电专属或地区受限视频，能否下载取决于所选浏览器登录账号是否有权限。平台可能允许读取标题和封面，但下载时仍返回权限错误；卡片会保留错误信息，可换有资格的 Firefox 账号后重试。
- YouTube 需要本机网络能够访问；平台限流或网络失败会显示错误并延后重试。关注设置里的「检查与下载登录态」默认使用 Firefox：订阅检查和后续 yt-dlp 下载共用同一个浏览器登录态，B 站被匿名请求拦截时成功率更高。Firefox 必须先登录对应平台；程序只在本机读取匹配域名的 Cookie，不上传 Cookie。Chrome / Edge 可能受 Cookie 加密限制，建议 Firefox。公开视频也可以选择「不使用」。

#### 关注更新的登录态

首次添加 B 站或 YouTube 关注时，建议先在 Firefox 登录对应网站，再选择「Firefox（推荐，检查和下载共用）」。程序会读取本机 Firefox 配置中的 `cookies.sqlite`，复制后只提取对应域名的 Cookie，用于订阅检查和视频下载。Firefox 正在运行时通常也可以读取；如果读取失败，完全退出 Firefox 后再点“检查”。

Cookie 失效、平台风控或网络限制仍可能导致 HTTP 412、-352 等错误。这表示平台拒绝了本次请求，不代表频道链接格式错误；稍后重试或在 Firefox 中重新登录即可。

### 编辑器

- **预览**：视频上方实时叠加字幕，字体/颜色/描边/字号/底边距所见即所得
- **拆分**：双击文本框进入编辑，光标放到目标位置，`Ctrl+Enter` 在最近词边界拆分（词级时间戳保证音画对齐）
- **合并**：段尾 ⇊ 按钮合并下一段
- **校对原稿**：跑 LLM 两遍校对，结果以 diff 弹窗逐条确认后应用
- **校对译文**：对照源稿与当前译文，建议可勾选后一键应用
- **翻译**：整片翻译或重译；结构变更（拆分/合并）过的段落会标记"需重译"
- **切片**：选时间段导出 MP4 片段（含烧录字幕）
- **烧录**：整片 ASS 烧录，音频重编码 AAC 192k

### 设置（任务详情）

- **样式**：字体（15 种）、字幕颜色、描边颜色、字号、底边距、说话人前缀开关、配色模式（统一/按说话人，单人自动回落统一色）
- **遮罩**：底部遮罩（模糊/纯色），高度、位置、透明度可调
- **说话人名称**：给 S01/S02… 起名，配合"说话人前缀"在预览/SRT/烧录中显示为 `名称: 字幕`
- **底部"保存修改"**：保存样式与字幕改动

---

## 配置文件（`config/`，已 gitignore，含密钥勿提交）

| 文件 | 内容 |
| --- | --- |
| `hotwords.json` | 全局 Whisper 热词词表（如 Neuro、Vedal 等专名）；上传/链接页维护，应用校对术语后自动沉淀 |
| `protected_terms.json` | 全局翻译保护词；HY-MT2 使用术语干预，AI 使用提示词，OPUS-MT 不使用 |
| `llm_profiles.json` | LLM API 配置（名称、Base URL、模型、API Key、是否关思考模式）；首页面板维护，含密钥，不要公开分发 |

## 目录结构与产物

```
├── moss_transcribe_diarize/     # Python 包（历史名，不改）
│   ├── app/
│   │   ├── server.py            # FastAPI 路由
│   │   ├── static/              # 前端三件套 index.html / style.css / app.js
│   │   ├── jobs.py              # 任务管理器、拆分/合并、字幕文件读写
│   │   ├── whisper_runner.py    # 转录（含缺口恢复）
│   │   ├── downloader.py        # yt-dlp 下载
│   │   ├── vocal_separator.py   # demucs 人声分离
│   │   ├── speaker_labeler.py   # pyannote 说话人分离
│   │   ├── proofreader.py       # LLM 两遍校对
│   │   ├── text_translator.py   # 翻译（openai/ollama）
│   │   ├── hy_mt_translator.py  # 本地 HY-MT2（按需拉起 llama-server）
│   │   ├── local_mt_translator.py # 本地 Opus-MT（CTranslate2）
│   │   ├── clips.py / ffmpeg.py # 切片与烧录
│   │   └── llm_profiles.py      # LLM 配置存储
│   ├── subtitle/                # 数据模型、SRT/ASS/文本导出、后处理
│   └── transcript_parser.py     # whisper 原始输出解析
├── tests/                       # Python 与浏览器回归测试
├── config/                      # 运行时配置（不入库）
├── models/                      # 本地模型（不入库）
├── tools/                       # ffmpeg / yt-dlp 可执行（不入库）
└── runs/<job_id>/               # 每个任务的产物（不入库）
    ├── input.*                  # 原始媒体
    ├── segments.json            # 字幕数据（含词级时间戳、翻译）
    ├── segments.source.json     # 翻译前源稿备份（结构同步，保证重译一致）
    ├── subtitle.srt             # SRT（供参考/外部编辑回同步）
    ├── subtitle.ass             # ASS（烧录用）
    ├── raw_transcript.txt       # 原始转录文本
    └── output.mp4               # 烧录成品（如有）
```

> 外部编辑提示：直接改 `runs/<job>/subtitle.srt` 会被网页检测到并同步回 `segments.json`（应用自己写出的文件不会误触发）。

---

## 字幕处理管线细节

1. **转录**：whisper 输出词级时间戳；长音频缺口自动恢复（时间偏移修正）
2. **重组**：`regroup_sentences_from_words` 按词级时间戳重建句子——标点软切、碎段合并、长段拆分，避免"半句话"断轴
3. **说话人归属**：pyannote 吃 demucs 人声轨产出说话人轮次，跨轮次段落按词级时间戳逐词归属
4. **校对**（翻译前）：Pass1 滑窗局部修正建议；Pass2 全片术语分析；查看后应用，费用由模型和文本量决定
5. **翻译**：结构变更的段落自动标记重译；源稿备份保证重译后结构一致

## 开发

```powershell
uv pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m ruff check moss_transcribe_diarize tests
npm ci
npx playwright install chromium  # 首次运行一次
npm run test:frontend
```

前端改动直接编辑 `moss_transcribe_diarize/app/static/` 下的文件（服务端 no-store，刷新即生效）。Playwright 测试使用固定 API 夹具，不加载模型或调用外部服务。CI（GitHub Actions）跑 ruff、pytest 和 Chromium 前端流程测试。

## 常见问题

- **GPU 没用上 / 想纯 CPU 跑**：手动启动传 `--device cuda --dtype float16`（N 卡）或 `--device cpu --dtype int8`（CPU）；`auto` 会自动探测。CPU 能跑但慢很多。
- **pyannote 报 401/403**：`speaker-diarization-3.1` 是 gated 模型——先在 HuggingFace 模型页登录并同意协议，再通过 `--hf-token` 或 `HF_TOKEN` 提供令牌；也可以把模型本地化到 `models/pyannote-speaker-diarization-local`，用 `--pyannote-model` 指向本地目录。
- **URL 下载失败 / 读不到浏览器 cookies**：默认使用 Firefox 的 cookies（Chrome/Edge 的 App-Bound Encryption 会导致 yt-dlp 读不到），可在首页"下载 Cookie"处换浏览器来源。
- **报 ffmpeg 不可用**：按"外部工具"一节放置便携版到 `tools/ffmpeg/`，或把 `ffmpeg` / `ffprobe` 装进系统 PATH。
- **模型下载慢 / 失败**：`start.bat` 已默认走 `hf-mirror.com` 镜像并调大超时；手动启动时自行设置 `HF_ENDPOINT` 等环境变量（见上表）。
- **端口被占用**：`mtd-subtitle-web --port 8080` 换端口启动。
- **想完全离线使用**：把用到的模型全部本地化到 `models/`，设 `HF_HUB_OFFLINE=1`。
- **给别人打包**：见“分发给别人”一节。当前推荐代码/工具与模型分包，目标电脑重建环境；直接复制开发机 `.venv` 不可靠，真正的免安装 runtime 整合包尚未实现。

## 已知限制

- 说话人归属在快速抢话、相似音色等场景可能出错，算法生成的标签数量不等于真实人数；现有回归结果不能当作所有素材的准确率保证
- 热词只对近音错听有效（如 Nero→Neuro），远距错听救不回来
- Windows 优先；其他平台未测试

## 致谢与许可

本项目基于 [OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 二次开发（包名 `moss_transcribe_diarize` 保留上游命名），在其转录/字幕骨架上重构并扩展了本地字幕工作台的大部分功能。遵循 [Apache-2.0](LICENSE) 协议开源。

依赖的关键开源组件：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（转录）、[pyannote.audio](https://github.com/pyannote/pyannote-audio)（说话人分离）、[demucs](https://github.com/adefossez/demucs)（人声分离）、[yt-dlp](https://github.com/yt-dlp/yt-dlp)（下载）、[FFmpeg](https://ffmpeg.org)（切片/烧录）。
