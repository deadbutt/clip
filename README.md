# 蝶殇工作台

本项目现在转向本地视频字幕与切片工作台：默认使用 Whisper/faster-whisper 转写，转写后进入说话人标记、字幕校对、中文翻译、精华片段选择和带字幕 MP4 导出。

旧的 MOSS 端到端大模型路线不再作为默认工作流；vLLM 保留给后续翻译、摘要、标题生成等文本模型任务。

## 启动

```bat
start.bat
```

启动后打开：

```text
http://127.0.0.1:7860
```

如果要启用英译中和 AI 精华精选，使用：

```bat
start_vllm.bat
```

首次运行会在可见的 WSL 窗口下载并启动 `Qwen/Qwen2.5-3B-Instruct-AWQ`。这是适合 8GB 显存的 4-bit 版本，同时提供英译中和候选片段语义排序。接口默认地址：

```text
http://127.0.0.1:8000/v1
```

## 推荐流程

1. 上传视频或音频。
2. 使用 Whisper 转写；8GB 显存建议先用 `small`，质量不够再试 `medium`。
3. 在字幕表和原片时间轴里人工校对文字与边界。
4. 需要中文时点击顶部“英译中”；翻译前会自动备份英文底稿。
5. 点击顶部“精华切片”：`AI 精选` 使用 Qwen 语义排序，`规则粗筛` 只是无模型候选。
6. 回看原片并调整绝对起止时间，再导出带字幕 MP4、片内 SRT 和原片时间映射 JSON。

## 更强的说话人分离

默认轻量 `cluster` 不需要额外模型，速度快，但多人节目只能当作粗稿。更强方案是 `pyannote`：

```bat
uv pip install -e ".[diarization]"
```

pyannote 的常用模型是 `pyannote/speaker-diarization-3.1`，需要 Hugging Face token，并且需要在 Hugging Face 页面接受模型条款。启动时传入：

```bat
.venv\Scripts\mtd-subtitle-web.exe --backend whisper --model small --diarization-backend pyannote --speaker-count 4 --hf-token YOUR_TOKEN
```

也可以设置环境变量 `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN`。没有安装 pyannote 或 token 不可用时，`auto` 会回退到 `cluster`；如果明确选择 `pyannote`，失败原因会显示在任务信息里。

## 命令行转写

```bat
.venv\Scripts\mtd-subtitle.exe "D:\path\to\input.mp4" --backend whisper --model small --device auto --dtype auto --diarization-backend auto --speaker-count 4
```

跳过说话人标记：

```bat
.venv\Scripts\mtd-subtitle.exe "D:\path\to\input.mp4" --backend whisper --model small --no-speaker-labeling
```

## 现实边界

Whisper 负责“听清楚并给时间段”，不是专业说话人分离模型。多人、重叠说话、笑声、背景音乐、同声线都会让自动分离变差。当前设计把自动结果当作第一稿，然后在工作台里人工校正和导出，这是比押注一个大模型端到端完成更靠谱的路线。
