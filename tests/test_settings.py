from config.settings import Settings


def test_settings_provider_normalization():
    s = Settings(
        asr_provider="DEEPGRAM",  # type: ignore
        llm_provider="OPENAI",  # type: ignore
    )
    assert s.asr_provider == "deepgram"
    assert s.llm_provider == "openai"


def test_settings_defaults():
    s = Settings()
    assert s.app_env in ("development", "production", "test")
    assert s.scratch_dir is not None
