__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import pytest
from executor.audioclip_text import AudioCLIPTextEncoder
from jina import Document, DocumentArray, Flow

_EMBEDDING_DIM = 1024


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [Document(text='just some random text here') for _ in range(50)]
    )
    with Flow(return_results=True).add(uses=AudioCLIPTextEncoder) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert len(resp) == 50
    for r in resp.embeddings:
        assert r.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu(build_docker_image_gpu: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'pea',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus',
                'all',
                '--uses-with',
                'device:"cuda"',
                'download_model:True',
            ],
            timeout=30,
            check=True,
        )
