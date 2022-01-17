import pytest
from jina import Document, DocumentArray

from transformer_qa import TransformerQAExecutor


@pytest.fixture
def docs():
    paragraphs = [
        'Document is basic data type in Jina. Flow ties Executors in a pipeline. ',
        'Jina supports Windows. Jina supports UNIX. ',
        'Jina supports Kubernetes. Also supports GPU. ',
        'Executor is how Jina processes Documents. DocumentArrayMemmap stores data on-disk.',
    ]

    questions = [
        'What is Document in Jina?',
        'Does Jina work on Windows?',
        'Does Jina support GPU?',
        'What is Executor?',
    ]

    expected = [
        'basic data type',
        'Jina supports Windows',
        'Also supports GPU',
        'how Jina processes Documents',
    ]

    docs = DocumentArray(
        [
            Document(text=p, tags={'question': q, 'expected': e})
            for p, q, e in zip(paragraphs, questions, expected)
        ]
    )
    return DocumentArray(
        Document(
            text='\n'.join(paragraphs),
            chunks=docs,
            tags={'question': questions[-1], 'expected': expected[-1]},
        )
    )


@pytest.mark.gpu
@pytest.mark.parametrize('traversal_paths', ['r', 'r,c'])
def test_search_gpu(docs, traversal_paths):
    # this only runs when we add gpu flag in pytest command i.e. "pytest --gpu ..."
    qa = TransformerQAExecutor(device='cuda', default_traversal_paths=traversal_paths)
    _test_search(docs, qa, traversal_paths)


@pytest.mark.parametrize('traversal_paths', ['r', 'r,c'])
def test_search(docs, traversal_paths):
    qa = TransformerQAExecutor(default_traversal_paths=traversal_paths)
    _test_search(docs, qa, traversal_paths)


def _test_search(docs, qa, traversal_paths):
    qa.generate(docs)
    for doc in docs.traverse_flat(traversal_paths):
        assert doc.matches[0].text == doc.tags['expected']
