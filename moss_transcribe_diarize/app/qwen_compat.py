# -*- coding: utf-8 -*-
"""qwen_asr 与 transformers 5.x 的运行时兼容垫片。

qwen-asr 0.0.6 的内置模型代码按 transformers 4.57.x 的 API 写成，在
transformers 5.x 下有两处不兼容，在导入 qwen_asr 之前打本垫片即可，
无需改动任何第三方包文件：

1. ``@check_model_inputs()`` 带括号的装饰器用法在 5.x 中改为直接作用于函数；
2. ``AutoConfig/AutoModel/AutoProcessor.register("qwen3_asr", ...)`` 在 5.x 中
   因原生已注册同名 model_type 而报错，且必须让 qwen_asr 自己的实现覆盖
   原生注册（原生 Qwen3ASR 类是给新一代 checkpoint 写的，权重结构对不上）。
"""

from __future__ import annotations

_PATCHED = False


def patch_check_model_inputs() -> None:
    """让 transformers 5.x 的 check_model_inputs 兼容 4.x 的带括号用法。"""
    import inspect

    import transformers.utils.generic as generic

    real = getattr(generic, "check_model_inputs", None)
    if real is None:
        return
    try:
        if "func" not in inspect.signature(real).parameters:
            return  # 已经是 4.x 风格，无需处理
    except (TypeError, ValueError):
        return

    def _compat(*args, **kwargs):  # noqa: ANN002, ANN003
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # 新式用法: @check_model_inputs
            return real(args[0])

        # 旧式用法: @check_model_inputs(...) -> 返回装饰器
        def decorator(func):
            return real(func)

        return decorator

    generic.check_model_inputs = _compat


def patch_auto_register_exist_ok() -> None:
    """让 Auto*.register 强制 exist_ok=True，允许 qwen_asr 覆盖原生注册。"""
    from transformers import AutoConfig, AutoModel, AutoProcessor

    for owner, name in (
        (AutoConfig, "register"),
        (AutoModel, "register"),
        (AutoProcessor, "register"),
    ):
        original = getattr(owner, name)
        try:
            import inspect

            if "exist_ok" not in inspect.signature(original).parameters:
                continue
        except (TypeError, ValueError):
            continue

        def make_forward(orig):  # noqa: ANN001, ANN202
            def forward(*args, **kwargs):  # noqa: ANN002, ANN003
                kwargs["exist_ok"] = True
                return orig(*args, **kwargs)

            return forward

        setattr(owner, name, make_forward(original))


def patch_qwen3_asr_config() -> None:
    """让 qwen_asr 的 config 类兼容 5.x 的两类行为变化。

    1. transformers 5.x 会在 ``super().__init__()`` 内部调用
       ``get_text_config()`` 做 token id 校验，而 qwen_asr 的实现里
       ``thinker_config`` 要到 ``super().__init__()`` 之后才赋值，
       构造期访问会 AttributeError。让它在未就绪时返回 self，
       校验因拿不到 vocab_size 自然跳过，不影响最终配置。

    2. 4.x 的 PretrainedConfig 对未定义属性返回 None，5.x 改为抛
       AttributeError，而 qwen_asr 的建模代码依赖 ``config.pad_token_id
       is not None`` 这类旧语义。这里仅对白名单属性（token id、
       timestamp 参数）恢复宽松访问，避免干扰 5.x 的严格校验器。
    """
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRThinkerConfig,
    )

    if getattr(Qwen3ASRConfig, "_mtd_patched", False):
        return

    def _safe_get_text_config(self, decoder=False):  # noqa: ANN001, ANN202
        thinker = getattr(self, "thinker_config", None)
        if thinker is None:
            return self
        return thinker.get_text_config()

    def _lenient_getattr(self, key):  # noqa: ANN001, ANN202
        if key.endswith("_token_id") or key == "timestamp_segment_time":
            return None
        raise AttributeError(f"{type(self).__name__} has no attribute {key!r}")

    Qwen3ASRConfig.get_text_config = _safe_get_text_config
    Qwen3ASRConfig.__getattr__ = _lenient_getattr
    Qwen3ASRThinkerConfig.__getattr__ = _lenient_getattr
    Qwen3ASRAudioEncoderConfig.__getattr__ = _lenient_getattr
    Qwen3ASRConfig._mtd_patched = True


def _default_rope_params(config, device=None):  # noqa: ANN001, ANN202
    """4.57.x 的默认 ROPE 频率计算（base^(2i/d) 倒数）。"""
    import torch

    head_dim = getattr(config, "head_dim", None)
    dim = head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0


def patch_rope_default() -> None:
    """补回 5.x 移除的 ROPE 'default' 初始化函数。

    qwen_asr 的 rotary embedding 通过 ``ROPE_INIT_FUNCTIONS[rope_type]``
    取初始化函数，config 的 rope_scaling.rope_type 为 "default"，
    而 5.x 的表里已没有这个键（默认路径被内联进基类了）。
    同时给 qwen_asr 的 rotary 类补上 5.x 初始化钩子会调用的
    ``compute_default_rope_parameters`` 方法。
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = lambda config=None, device=None, **kw: _default_rope_params(
            config, device
        )

    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRThinkerTextRotaryEmbedding,
    )

    if not hasattr(Qwen3ASRThinkerTextRotaryEmbedding, "compute_default_rope_parameters"):

        def _compute_default_rope_parameters(self, config=None):  # noqa: ANN001, ANN202
            return _default_rope_params(config or self.config)

        Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters = (
            _compute_default_rope_parameters
        )


def patch_generate_mask_key() -> None:
    """把 5.x processor 输出的 input_features_mask 映射回 4.x 的键名。

    AutoProcessor 懒加载按类名解析到 transformers 原生 Qwen3ASRProcessor，
    其输出把音频有效帧掩码命名为 ``input_features_mask``；qwen_asr 的
    建模代码与 generate 白名单只认 ``feature_attention_mask``。
    """
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRForConditionalGeneration,
    )

    if getattr(Qwen3ASRForConditionalGeneration, "_mask_key_patched", False):
        return

    _orig_generate = Qwen3ASRForConditionalGeneration.generate

    def _generate(self, input_ids=None, max_new_tokens=4096, eos_token_id=(151645, 151643), **kwargs):  # noqa: ANN001, ANN002, ANN003
        if "input_features_mask" in kwargs:
            kwargs.setdefault("feature_attention_mask", kwargs.pop("input_features_mask"))
        return _orig_generate(
            self, input_ids=input_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, **kwargs
        )

    Qwen3ASRForConditionalGeneration.generate = _generate
    Qwen3ASRForConditionalGeneration._mask_key_patched = True


def patch_prepare_inputs_cache_position() -> None:
    """在 prefill 首步为 qwen_asr 的 prepare_inputs_for_generation 合成 cache_position。

    5.x 的 _prefill 不再把 cache_position 放进 model_kwargs（基类内部
    自行计算），而 qwen_asr 的 override 直接下标访问 ``cache_position[0]``。
    """
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
    )

    if getattr(Qwen3ASRThinkerForConditionalGeneration, "_cache_pos_patched", False):
        return

    _orig_prepare = Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation

    def _prepare_inputs_for_generation(  # noqa: ANN001, ANN202
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        input_features=None,
        feature_attention_mask=None,
        **kwargs,
    ):
        if cache_position is None:
            import torch

            past_length = 0
            if past_key_values is not None:
                try:
                    past_length = past_key_values.get_seq_length()
                except Exception:  # noqa: BLE001 - 不同 cache 实现的容错
                    past_length = 0
            cache_position = torch.arange(
                past_length, input_ids.shape[1], device=input_ids.device
            )
        return _orig_prepare(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )

    Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation = (
        _prepare_inputs_for_generation
    )
    Qwen3ASRThinkerForConditionalGeneration._cache_pos_patched = True


def patch_create_causal_mask() -> None:
    """让 qwen_asr 的 create_causal_mask 调用兼容 5.x 签名。

    5.x 的 ``create_causal_mask`` 参数名是 ``inputs_embeds``（有 s），
    且已从签名移除 ``cache_position``（docstring 标 deprecated/unused）。
    qwen_asr 的建模代码仍按 4.x 调用：``input_embeds``（没 s）
    + ``cache_position``。这里包一层做参数名映射并丢弃 cache_position。
    """
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as mod

    if getattr(mod, "_causal_mask_patched", False):
        return

    _orig = mod.create_causal_mask

    def _compat(*args, **kwargs):  # noqa: ANN002, ANN003
        # 4.x input_embeds (没 s) -> 5.x inputs_embeds (有 s)
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        # 5.x 已从签名移除 cache_position（deprecated unused）
        kwargs.pop("cache_position", None)
        return _orig(*args, **kwargs)

    mod.create_causal_mask = _compat
    mod._causal_mask_patched = True


def patch_thinker_forward_mask_key() -> None:
    """覆盖 forced aligner 路径 thinker.forward 收到的 mask 键名。

    patch_generate_mask_key 只映射顶层 ``generate`` 入口的 mask 键；
    而 forced aligner (qwen3_forced_aligner.py) 直接
    ``self.model.thinker(**inputs)``，inputs 是 5.x processor 输出
    （键名 ``input_features_mask``），thinker.forward 形参认
    ``feature_attention_mask``，多余键落进 ``**kwargs`` 导致
    ``feature_attention_mask`` 为 None，get_audio_features 在
    ``feature_attention_mask.sum(-1)`` 处抛 AttributeError。

    这里在 thinker.forward 入口做一次键名映射。守卫条件
    ``feature_attention_mask is None`` 保证 generate 路径（该参数已有值）
    不受影响。
    """
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
    )

    if getattr(Qwen3ASRThinkerForConditionalGeneration, "_thinker_mask_patched", False):
        return

    _orig_forward = Qwen3ASRThinkerForConditionalGeneration.forward

    def _forward(  # noqa: ANN001, ANN202
        self,
        input_ids=None,
        input_features=None,
        attention_mask=None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        rope_deltas=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if feature_attention_mask is None and "input_features_mask" in kwargs:
            feature_attention_mask = kwargs.pop("input_features_mask")
        return _orig_forward(
            self,
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    Qwen3ASRThinkerForConditionalGeneration.forward = _forward
    Qwen3ASRThinkerForConditionalGeneration._thinker_mask_patched = True


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    patch_check_model_inputs()
    patch_auto_register_exist_ok()
    patch_rope_default()
    _PATCHED = True


def import_qwen_asr():
    """导入 qwen_asr 官方包（先打垫片）。"""
    apply()
    import qwen_asr  # noqa: F401

    patch_qwen3_asr_config()
    patch_generate_mask_key()
    patch_prepare_inputs_cache_position()
    patch_create_causal_mask()
    patch_thinker_forward_mask_key()
    return qwen_asr
