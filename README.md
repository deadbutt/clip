# 蝶殇工作台 (DieShang Workbench)

基于 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 二次开发的本地视频字幕工作台:下载 → 转录 → 说话人分离 → 翻译 → LLM 校对 → 字幕烧录/切片,全流程在本机完成。

## 功能

- **视频/音频下载**:通过 yt-dlp 拉取在线视频,支持浏览器 Cookie 与字幕抓取
- **语音转录**:faster-whisper(local large-v3-turbo / medium),词级时间戳,热词保护(`config/hotwords.json`)
- **人声分离与说话人分离**:demucs(htdemucs)人声增强 + pyannote speaker-diarization 3.1(本地模型)
- **翻译**:OpenAI 兼容 API(支持多 profile 切换)或本地 Opus-MT(ct2 int8)英→中
- **LLM 字幕校对**:两遍架构——小窗口 1:1 局部润色 + 全片只读结构分析,LLM 永不重写全文
- **字幕导出**:SRT / ASS(说话人配色、遮罩模糊)/ 纯文本,ffmpeg 烧录,片段切片导出 MP4
- **Web 界面**:实时进度、波形拖动、字幕编辑、SRT 直接同步回写

## 安装

需要 Python 3.10+、[uv](https://docs.astral.sh/uv/)、ffmpeg(或放入 `tools/ffmpeg/`)。

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[torch-runtime,diarization]"
```

首次使用按需运行根目录的 `download_*.bat` 下载本地模型(whisper / diarization / 翻译模型)。

## 启动

- Web 界面:双击 `start.bat`,或 `mtd-subtitle-web`
- 命令行:`mtd-subtitle --help`

## 目录约定

| 路径 | 用途 |
| --- | --- |
| `runs/` | 每个任务的媒体、字幕、JSON 产物(不入库) |
| `config/` | 热词表与 LLM profile(**含 API key,勿提交**) |
| `models/` | 本地模型文件(不入库) |
| `tests/` | 单元测试:`pytest tests/` |

## License

参见 [LICENSE](LICENSE)。
