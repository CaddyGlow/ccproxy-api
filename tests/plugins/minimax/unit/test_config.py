"""Configuration tests for the MiniMax provider plugin."""

import pytest

from ccproxy.plugins.minimax.config import MINIMAX_REGION_ENDPOINTS, MiniMaxConfig
from ccproxy.plugins.minimax.model_defaults import (
    DEFAULT_MINIMAX_MODEL_CARDS,
    DEFAULT_MINIMAX_MODEL_MAPPINGS,
)


@pytest.mark.minimax
@pytest.mark.unit
def test_default_region_is_global() -> None:
    config = MiniMaxConfig()
    assert config.name == "minimax"
    assert config.region == "global_en"
    assert config.base_url == "https://api.minimax.io/v1"
    assert config.anthropic_base_url == "https://api.minimax.io/anthropic"
    assert config.auth_type == "api_key"


@pytest.mark.minimax
@pytest.mark.unit
def test_cn_region_resolves_regional_endpoints() -> None:
    config = MiniMaxConfig(region="cn_zh")
    assert config.base_url == MINIMAX_REGION_ENDPOINTS["cn_zh"]["openai_base_url"]
    assert (
        config.anthropic_base_url
        == MINIMAX_REGION_ENDPOINTS["cn_zh"]["anthropic_base_url"]
    )
    assert config.docs_root == MINIMAX_REGION_ENDPOINTS["cn_zh"]["docs_root"]


@pytest.mark.minimax
@pytest.mark.unit
def test_explicit_base_url_overrides_region() -> None:
    config = MiniMaxConfig(region="cn_zh", base_url="https://proxy.example/v1")
    assert config.base_url == "https://proxy.example/v1"
    # Unset fields still follow the region.
    assert (
        config.anthropic_base_url
        == MINIMAX_REGION_ENDPOINTS["cn_zh"]["anthropic_base_url"]
    )


@pytest.mark.minimax
@pytest.mark.unit
def test_default_models_present() -> None:
    config = MiniMaxConfig()
    models = {card.id: card for card in config.models_endpoint}
    assert list(models) == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert {card.id for card in DEFAULT_MINIMAX_MODEL_CARDS} == set(models)
    assert len(DEFAULT_MINIMAX_MODEL_MAPPINGS) >= 1

    m3 = models["MiniMax-M3"].model_dump()
    assert m3["context_window"] == 1_000_000
    assert m3["pricing_usd_per_million_tokens"] == {
        "input": 0.3,
        "output": 1.2,
        "cache_read": 0.06,
        "cache_write": None,
    }
    assert m3["input_modalities"] == ["text", "image", "video"]
    assert m3["thinking"] == ["adaptive", "disabled"]

    assert m3["pricing_context_tiers"][1] == {
        "context": ">512k",
        "input": 0.6,
        "output": 2.4,
        "cache_read": 0.12,
    }
    flash = models["MiniMax-M3.1-Flash"].model_dump()
    assert flash["context_window"] == 1_000_000
    assert flash["max_output_tokens"] == 524_288
    assert flash["pricing_usd_per_million_tokens"] == {
        "input": 0.15,
        "output": 0.60,
        "cache_read": 0.015,
        "cache_write": None,
    }
    assert flash["input_modalities"] == ["text", "image", "video"]
    assert flash["thinking"] == ["adaptive"]
    assert flash["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert flash["default_reasoning_effort"] == "max"


@pytest.mark.minimax
@pytest.mark.unit
def test_explicit_model_configuration_preserved() -> None:
    from ccproxy.models.provider import ModelCard, ModelMappingRule
    from ccproxy.utils.model_mapper import ModelMapper

    card = ModelCard(id="MiniMax-M2.7", created=1, owned_by="minimax")
    rules = [ModelMappingRule(match="minimax", target="MiniMax-M2.7")]
    config = MiniMaxConfig(models_endpoint=[card], model_mappings=rules)
    assert config.models_endpoint == [card]
    assert config.model_mappings == rules
    assert ModelMapper(config.model_mappings).map("minimax").mapped == "MiniMax-M2.7"
