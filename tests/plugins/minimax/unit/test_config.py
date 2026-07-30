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
    ids = {card.id for card in config.models_endpoint}
    assert ids == {"MiniMax-M3", "MiniMax-M2.7"}
    assert {card.id for card in DEFAULT_MINIMAX_MODEL_CARDS} == ids
    assert len(DEFAULT_MINIMAX_MODEL_MAPPINGS) >= 1
