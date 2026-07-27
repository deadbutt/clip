from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class SubtitleSegment:
    id: str
    start: float
    end: float
    speaker: str
    text: str

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
