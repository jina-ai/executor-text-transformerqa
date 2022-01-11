import pytest
from jina import Document, DocumentArray

from transformer_qa import TransformerQAExecutor


@pytest.fixture
def docs():
    paragraph = (
        'Document is basic data type in Jina. Flow ties Executors in a pipeline. '
        'Jina supports Windows. Jina supports UNIX. '
        'Jina supports Kubernetes. Also supports GPU. '
        'Executor is how Jina processes Documents. DocumentArrayMemmap stores data on-disk.'
    )

    docs = DocumentArray(
        [
            Document(
                text=paragraph,
                tags={
                    'question': 'What is Document in Jina?',
                    'expected': 'basic data type',
                },
            ),
            Document(
                text=paragraph,
                tags={
                    'question': 'Does Jina work on Windows',
                    'expected': 'Jina supports Windows',
                },
            ),
            Document(
                text=paragraph,
                tags={
                    'question': 'Does Jina support GPU',
                    'expected': 'Also supports GPU',
                },
            ),
            Document(
                text=paragraph,
                tags={
                    'question': 'What is Executor',
                    'expected': 'how Jina processes Documents',
                },
            ),
        ]
    )
    return docs


@pytest.mark.gpu
def test_search_gpu(docs):
    # this only runs when we add gpu flag in pytest command i.e. "pytest --gpu ..."
    qa = TransformerQAExecutor(device='cuda')
    _test_search(docs, qa)


def test_search(docs):
    qa = TransformerQAExecutor()
    _test_search(docs, qa)


def _test_search(docs, qa):
    qa.generate(docs)
    for doc in docs:
        assert doc.matches[0].text == doc.tags['expected']
