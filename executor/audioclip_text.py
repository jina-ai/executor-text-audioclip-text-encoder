__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import warnings
from typing import Optional

import torch
from jina import DocumentArray, Executor, requests
from jina.logging.predefined import default_logger

from .audio_clip.model import AudioCLIP


class AudioCLIPTextEncoder(Executor):
    """
    Encode text data with the AudioCLIP model
    """

    def __init__(
        self,
        model_path: str = '.cache/AudioCLIP-Full-Training.pt',
        tokenizer_path: str = '.cache/bpe_simple_vocab_16e6.txt.gz',
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        batch_size: int = 32,
        device: str = 'cpu',
        download_model: bool = False,
        *args,
        **kwargs
    ):
        """
        :param model_path: path to the pre-trained AudioCLIP model.
        :param access_paths: default traversal path (used if not specified in
            request's parameters)
        :param traversal_paths: please use access_paths
        :param batch_size: default batch size (used if not specified in
            request's parameters)
        :param device: device that the model is on (should be "cpu", "cuda" or
        "cuda:X",
            where X is the index of the GPU on the machine)
        :param download_model: whether to download the model at start-up
        """
        super().__init__(*args, **kwargs)
        default_logger.debug(f'torch has cuda available: {torch.cuda.is_available()}')

        if download_model:
            import os
            import subprocess

            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            script_name = 'scripts/download_full.sh'
            if 'Partial' in model_path:
                script_name = 'scripts/download_partial.sh'
            subprocess.call(['sh', script_name], cwd=root_path)

        default_logger.debug(f'loading model from {model_path}')
        default_logger.debug(f'loading tokenizer from {tokenizer_path}')
        default_logger.debug(f'using device: {device}')
        self.model = (
            AudioCLIP(
                pretrained=model_path,
                bpe_path=tokenizer_path,
            )
            .to(device)
            .eval()
        )
        default_logger.debug(f'model loaded, model defive: {self.model.device}')

        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths
        self.batch_size = batch_size

    @requests
    def encode(
        self,
        docs: Optional[DocumentArray] = None,
        parameters: dict = {},
        *args,
        **kwargs
    ) -> None:
        """
        Method to create embeddings for documents by encoding their text.

        :param docs: A document array with documents to create embeddings for. Only
        the
            documents that have the ``text`` attribute will get embeddings.
        :param parameters: A dictionary that contains parameters to control encoding.
            The accepted keys are ``access_paths`` and ``batch_size`` - in their
            absence their corresponding default values are used.
        """
        default_logger.debug(f'model loaded, model device: {self.model.device}')
        if not docs:
            return

        tpaths = parameters.get('access_paths', self.access_paths)
        batch_generator = DocumentArray(
            filter(lambda doc: len(doc.text) > 0, docs[tpaths])
        ).batch(
            batch_size=parameters.get('batch_size', self.batch_size),
        )

        with torch.inference_mode():
            for batch in batch_generator:
                embeddings = self.model.encode_text(text=[[doc.text] for doc in batch])
                embeddings = embeddings.cpu().numpy()

                for idx, doc in enumerate(batch):
                    doc.embedding = embeddings[idx]
