# 蝶殇工作台 (DieShang Workbench)

基于 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 二次开发的**本地视频字幕工作台**：

```
下载 → 转录 → 人声分离 → 说话人分离 → 字幕后处理 → LLM 校对 → 翻译 → 编辑 → 烧录 / 切片
```

全流程在本机完成，媒体与字幕不经过任何第三方服务器（仅转录/校对/翻译按需调用你配置的 LLM API）。

---

## 功能总览

| 环节 | 实现 | 说明 |
| --- | --- | --- |
| 视频/音频下载 | yt-dlp（`tools/yt-dlp/yt-dlp.exe`） | 默认 `best[height<=1080]` 保留视频供烧录；输出 MKV 容器；默认 `--cookies-from-browser firefox`；实时进度（百分比/速度/ETA） |
| 语音转录 | faster-whisper（large-v3-turbo / medium） | 词级时间戳；VAD 用 faster-whisper 原生默认；`hallucination_silence_threshold=2.0` 抑制静音幻觉；支持热词 |
| 人声分离 | demucs（htdemucs） | 与 whisper 并行执行；`has_background_audio` 门控（词间隙/语音响度比 > 0.12 才启用人声轨）；whisper 始终吃原始音频 |
| 说话人分离 | pyannote speaker-diarization-3.1 | auto 模式自由聚类（上限 4 人）+ 时长占比 <10% 杂簇收编；跨说话人段用词级时间戳归属；说话人数量可指定 1-10 |
| 字幕后处理 | 词级重组 | 从词级时间戳重新断句，逗号软切（优先标点边界，避免在动词/介词/固定短语中间硬切）；合并 ≤4 权重碎段、拆分 >11s 长段；剔除重复幻觉 |
| LLM 校对 | 两遍架构 | Pass1 滑动窗口 1:1 局部替换（自动应用）；Pass2 全片只读标注（术语表自动应用，合并/说话人建议人工审核）；LLM 永不重写全文 |
| 翻译 | OpenAI 兼容 API / Ollama / 本地 Opus-MT | 多 profile 随时切换；校对跑在翻译前的源稿上；拆分/合并后自动标记需重译段落 |
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
| `torch-runtime` | torch / torchaudio / librosa（转录、demucs 必需） |
| `diarization` | pyannote.audio（说话人分离必需） |
| `flash-attn` | FlashAttention（转录加速，需对应 CUDA 环境） |
| `dev` | pytest |

### 本地模型

首次使用按需运行根目录脚本下载模型到 `models/`（已 gitignore）：

| 脚本 | 模型 | 用途 |
| --- | --- | --- |
| `download_faster_whisper_large_v3_turbo.bat` | faster-whisper-large-v3-turbo | 转录主力模型（默认） |
| `download_faster_whisper_medium.bat` | faster-whisper-medium | 轻量备选 |
| `download_qwen3_asr.bat` | qwen3-asr-1.7b | 备选 ASR 后端（质量弱于 whisper，未进生产） |

pyannote 需要 HF Token（gated 模型），或提前本地化到 `models/pyannote-speaker-diarization-local`。
demucs（htdemucs）与 Opus-MT 翻译模型首次运行自动下载。

### 外部工具

- **ffmpeg**：放 `tools/ffmpeg/ffmpeg.exe`（项目强制使用该路径，不依赖 PATH）
- **yt-dlp**：放 `tools/yt-dlp/yt-dlp.exe`

---

## 启动

### Web 界面（推荐）

双击 `start.bat`，或：

```powershell
mtd-subtitle-web                # 默认 http://127.0.0.1:7860
mtd-subtitle-web --port 8080    # 自定义端口
```

常用参数：`--model`（whisper 模型路径）、`--device`、`--dtype`、`--language`（固定语言跳过检测）、`--beam-size`、`--diarization-backend`、`--speaker-count`、`--hf-token`、翻译相关 `--translator-*`。

### 命令行

```powershell
mtd-subtitle --help
```

核心参数与 Web 版一致，另支持 `--segments-input`（跳过转录直接吃已有 segments JSON）、`--no-speaker-labeling`、`--out-dir` 等。

---

## Web 界面指南

### 首页（上传）

- **文件上传**：本地音视频，多文件排队
- **URL 任务**：在线视频直链，先下载后转录，进度实时显示
- **AI 服务**（折叠面板）：配置 LLM API，可存多份（DeepSeek / Ollama / 任意 OpenAI 兼容），"启用"按钮切换当前使用的配置；"测试连通"一键验证
- **下载 Cookie**：默认 Firefox（Chrome/Edge 的 App-Bound Encryption 导致 yt-dlp 读不到）

### 编辑器

- **预览**：视频上方实时叠加字幕，字体/颜色/描边/字号/底边距所见即所得
- **拆分**：双击文本框进入编辑，光标放到目标位置，`Ctrl+Enter` 在最近词边界拆分（词级时间戳保证音画对齐）
- **合并**：段尾 ⇊ 按钮合并下一段
- **校对**：跑 LLM 两遍校对，结果以 diff 弹窗逐条确认后应用
- **翻译**：整片翻译或重译；结构变更（拆分/合并）过的段落会标记"需重译"
- **切片**：选时间段导出 MP4 片段（含烧录字幕）
- **烧录**：整片 ASS 烧录，音频重编码 AAC 192k

### 设置（任务详情）

- **样式**：字体（15 种）、字幕颜色、描边颜色、字号、底边距、说话人前缀开关、配色模式（统一/按说话人，单人自动回落统一色）
- **遮罩**：底部遮罩（模糊/纯色），高度、位置、透明度可调
- **说话人名称**：给 S01/S02… 起名，配合"说话人前缀"在预览/SRT/烧录中显示为 `名称: 字幕`
- **热词词表**：全局转录热词（建任务自动带上）；校对确认的术语自动沉淀进来
- **底部"保存修改"**：保存样式与字幕改动

---

## 配置文件（`config/`，已 gitignore，含密钥勿提交）

| 文件 | 内容 |
| --- | --- |
| `hotwords.json` | 全局热词词表（如 Neuro、Vedal 等专名）；网页设置里维护，校对术语自动沉淀 |
| `llm_profiles.json` | LLM API 配置（名称、Base URL、模型、API Key、是否关思考模式）；首页面板维护 |

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
│   │   ├── text_translator.py   # 翻译（openai/ollama/opus-mt）
│   │   ├── clips.py / ffmpeg.py # 切片与烧录
│   │   └── llm_profiles.py      # LLM 配置存储
│   ├── subtitle/                # 数据模型、SRT/ASS/文本导出、后处理
│   └── transcript_parser.py     # whisper 原始输出解析
├── tests/                       # pytest 测试（142 个）
├── config/                      # 运行时配置（不入库）
├── models/                      # 本地模型（不入库）
├── tools/                       # ffmpeg / yt-dlp 可执行（不入库）
└── runs/<job_id>/               # 每个任务的产物（不入库）
    ├── input.*                  # 原始媒体
    ├── segments.json            # 字幕数据（含词级时间戳、翻译）
    ├── source_segments.json     # 翻译前源稿备份（结构同步，保证重译一致）
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
4. **校对**（翻译前）：Pass1 滑窗 1:1 替换自动应用；Pass2 全片只读标注人工审核——实测约 ¥0.1 / 21 分钟（DeepSeek）
5. **翻译**：结构变更的段落自动标记重译；源稿备份保证重译后结构一致

## 开发

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

前端改动直接编辑 `moss_transcribe_diarize/app/static/` 下的文件（服务端 no-store，刷新即生效）。

## 已知限制

- 说话人归属在两人快速抢话的边界处会有少量张冠李戴（GT 实测约 81% 准确率，人数判断 1-3 人全对）
- 热词只对近音错听有效（如 Nero→Neuro），远距错听救不回来
- Windows 优先；其他平台未测试

## License

参见 [LICENSE](LICENSE)。
