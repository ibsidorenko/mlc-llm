import argparse
import os
import pytest

from mlc_serve.engine import get_engine_config
from mlc_serve.model.paged_cache_model import PagedCacheModelModule
from mlc_serve.model.base import get_model_artifact_config
from mlc_serve.model.tvm_model import init_tvm_model


def _test_insufficient_cache_blocks_fail(artifact_path):
    model_artifact_path = os.path.join(artifact_path, "codellama-13b-instruct-hf-q0f16")

    if not os.path.exists(os.path.join(model_artifact_path)):
        return

    def try_init(max_num_seqs):
        engine_config = get_engine_config(
            {
                "use_staging_engine": False,
                "max_num_batched_tokens": 16384 * max_num_seqs,
                "min_decode_steps": 12,
                "max_decode_steps": 16,
            }
        )

        PagedCacheModelModule(
            model_artifact_path=model_artifact_path,
            engine_config=engine_config,
        )

    with pytest.raises(RuntimeError) as e_info:
        # This test assumes that 80GB VRAM is available.
        try_init(2)

    assert "Try reducing" in str(e_info.value)


def _test_catch_cache_alloc_oom(artifact_path):
    model_artifact_path = os.path.join(artifact_path, "llama-2-13b-chat-hf-q0f16")

    if not os.path.exists(os.path.join(model_artifact_path)):
        return

    model_artifact_config = get_model_artifact_config(model_artifact_path)

    engine_config = get_engine_config(
        {
            "max_num_batched_tokens": 40960
        }
    )

    with pytest.raises(RuntimeError) as e_info:
        init_tvm_model(model_artifact_config, engine_config)

    assert "Failed to allocate" in str(e_info.value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-path", type=str, default="dist")
    args = parser.parse_args()

    _test_insufficient_cache_blocks_fail(args.artifact_path)
    _test_catch_cache_alloc_oom(args.artifact_path)
