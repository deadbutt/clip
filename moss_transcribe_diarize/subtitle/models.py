from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SubtitleItem:
    """词级时间戳：转写引擎给出的最小对齐单元，拆分/切点估算的真源。"""

    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleItem":
        return cls(
            text=str(data.get("text") or ""),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
        )

    @classmethod
    def coerce(cls, value: Any) -> "SubtitleItem | None":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            item = cls.from_dict(value)
            return item if item.text else None
        return None


def coerce_subtitle_items(value: Any) -> list[SubtitleItem] | None:
    """把任意 payload 规整成 items 列表；无法解析时返回 None（视为缺失）。"""
    if not isinstance(value, list):
        return None
    items = [coerced for entry in value if (coerced := SubtitleItem.coerce(entry)) is not None]
    return items or None


@dataclass(slots=True)
class SubtitleSegment:
    id: str
    start: float
    end: float
    speaker: str
    text: str
    items: list[SubtitleItem] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, fallback_id: str | None = None) -> "SubtitleSegment":
        return cls(
            id=str(data.get("id") or fallback_id or ""),
            start=float(data["start"]),
            end=float(data["end"]),
            speaker=str(data.get("speaker") or "S00"),
            text=str(data.get("text") or ""),
            items=coerce_subtitle_items(data.get("items")),
        )


@dataclass(slots=True)
class SubtitleStyle:
    font_name: str = "Noto Sans CJK SC"
    font_size: int | None = None
    alignment: int = 2
    margin_v: int = 56
    show_speaker: bool = False
    speaker_colors: bool = False
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    back_color: str = "&H64000000"
    outline: int = 3
    shadow: int = 1
    speaker_names: dict[str, str] | None = None
    # 说话人 → 自定义字幕颜色(ASS &H00BBGGRR),未指定的说话人用调色板默认色。
    speaker_color_overrides: dict[str, str] | None = None
    mask_enabled: bool = False
    mask_mode: str = "blur"
    mask_height: int = 120
    mask_margin_v: int = 0
    mask_opacity: float = 0.82
    mask_blur: int = 24

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SubtitleStyle":
        if not data:
            return cls()
        style = cls()
        for field in cls.__dataclass_fields__:
            if field not in data:
                continue
            value = data[field]
            if field == "font_size":
                setattr(style, field, None if value in ("", None) else int(value))
            elif field == "speaker_names":
                if isinstance(value, dict):
                    names = {str(key): str(name).strip() for key, name in value.items() if str(name).strip()}
                    setattr(style, field, names or None)
            elif field == "speaker_color_overrides":
                if isinstance(value, dict):
                    overrides = {
                        str(key): str(color)
                        for key, color in value.items()
                        if str(color).startswith("&H") and len(str(color)) == 10
                    }
                    setattr(style, field, overrides or None)
            elif field in {"alignment", "margin_v", "outline", "shadow"}:
                setattr(style, field, int(value))
            elif field in {"mask_height", "mask_margin_v"}:
                setattr(style, field, int(value))
            elif field == "mask_opacity":
                setattr(style, field, float(value))
            elif field == "mask_blur":
                setattr(style, field, int(value))
            elif field in {"show_speaker", "speaker_colors", "mask_enabled"}:
                setattr(style, field, bool(value))
            elif field == "mask_mode":
                mode = str(value)
                setattr(style, field, mode if mode in {"blur", "bar"} else "blur")
            else:
                setattr(style, field, str(value))
        return style

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
