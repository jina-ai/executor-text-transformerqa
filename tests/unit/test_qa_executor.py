import pytest
from jina import Document, DocumentArray

from transformer_qa import TransformerQAExecutor


@pytest.fixture
def docs():
    docs = DocumentArray(
        [
            Document(
                text='What is Document in Jina', tags={'expected': 'basic data type'}
            ),
            Document(
                text='Does Jina work on Windows',
                tags={'expected': 'Jina supports Windows'},
            ),
            Document(
                text='Does Jina support GPU', tags={'expected': 'Also supports GPU'}
            ),
            Document(
                text='What is Executor',
                tags={'expected': 'how Jina processes Documents'},
            ),
        ]
    )
    matches = DocumentArray(
        [
            Document(
                text='Document is basic data type in Jina. Flow ties Executors in a pipeline.',
                scores={'cosine': 1, 'uri_cosine': 1},
            ),
            Document(
                text='Jina supports Windows. Jina supports UNIX.',
                scores={'cosine': 1, 'uri_cosine': 1},
            ),
            Document(
                text='Jina supports Kubernetes. Also supports GPU',
                scores={'cosine': 1, 'uri_cosine': 1},
            ),
            Document(
                text='Executor is how Jina processes Documents. DocumentArrayMemmap stores data on-disk.',
                scores={'cosine': 1, 'uri_cosine': 1},
            ),
        ]
    )
    for doc in docs:
        doc.matches.extend(matches)
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


def test_concatenate_matches(docs):
    qa = TransformerQAExecutor()
    concatenated_matches, span_range_to_match = qa._concatenate_matches(docs[0].matches)

    for (span_min, span_max, idx) in span_range_to_match:
        assert (
            docs[0].matches[idx].text == concatenated_matches[span_min : span_max - 1]
        )
