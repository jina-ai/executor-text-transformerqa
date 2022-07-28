from enum import IntEnum
from typing import List, Optional, Dict
from warnings import warn
from transformers import pipeline

from jina import Document, DocumentArray, Executor, requests


class Device(IntEnum):
    cpu = -1
    cuda = 0


class TransformerQAExecutor(Executor):
    def __init__(
        self,
        device: str = 'cpu',
        model_name: str = 'bert-large-uncased-whole-word-masking-finetuned-squad',
        tokenizer_name: Optional[str] = None,
        default_limit: int = 10,
        default_access_paths: str = 'r',
        default_traversal_paths: Optional[str] = None,
        **kwargs
    ):
        """
        TransformerQAExecutor wraps a question-answering model from Hugging face.
        Given some questions and paragraphs/contexts, it extracts the relevant answers
        from the given paragraph/context.

        :param device: the device to use (either cpu or cuda).
        :param model_name: the name of the QA model to use
        :param tokenizer_name: the name of the tokenizer to use. If not provided, model_name is used.
        :param default_limit: the default value that the QA model uses to specify the number of answers to
            extract from the paragraphs. It is also possible to pass `limit` in parameters to the `generate`
            method to change the default.
        :param default_access_paths: the default access path that QA model uses to traverse documents
        :param default_traversal_paths: please use default_access_paths
        """

        super().__init__(**kwargs)
        tokenizer_name = tokenizer_name or model_name
        self.default_limit = default_limit

        if default_traversal_paths is not None:
            self.default_access_paths = default_traversal_paths
            warn("'default_traversal_paths' will be deprecated in the future, please use 'default_access_paths'",
                 DeprecationWarning,
                 stacklevel=2)
        else:
            self.default_access_paths = default_access_paths

        self.model = pipeline(
            'question-answering',
            model=model_name,
            tokenizer=tokenizer_name,
            device=Device[device],
        )

    @requests(on='/search')
    def generate(
        self, docs: DocumentArray, parameters: Optional[Dict] = None, **kwargs
    ):
        """
        The method extracts the answers of a question from the given paragraph.

        :param docs: a list of documents where each document has `text` as the paragraph,
            and question in `doc.tags['question']`
        :param parameters: user can change the default limit by passing `limit` in `parameters`
            user can also change the default access path by passing `access_paths` in `parameters`
        """
        parameters = parameters or {}
        limit = parameters.get('limit', self.default_limit)
        access_paths = parameters.get(
            'access_paths', self.default_access_paths
        )

        for doc in docs.traverse_flat(access_paths):
            qa_input = [{'context': doc.text, 'question': doc.tags['question']}]
            qa_outputs = self.model(qa_input, limit=limit)
            if not isinstance(qa_outputs, list):
                qa_outputs = [qa_outputs]

            targets = [o['answer'] for o in qa_outputs]
            confidence_scores = [o['score'] for o in qa_outputs]

            doc.matches.clear()
            for (target, confidence_score,) in zip(
                targets,
                confidence_scores,
            ):
                match = Document(
                    text=target,
                    scores={
                        'confidence': confidence_score,
                    },
                )
                doc.matches.append(match)

            doc.matches.sort(lambda m: m.scores['confidence'].value, reverse=True)
