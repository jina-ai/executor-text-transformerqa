import pytest
from jina import Document, DocumentArray, Flow

from transformer_qa import TransformerQAExecutor


def test_flow():
    f = Flow().add(uses=TransformerQAExecutor)

    doc = Document(
        content='what is the color of an apple?',
        matches=[
            Document(text='Apple is a pome fruit that is red in color.'),
            Document(text='Banana is yellow.'),
        ],
    )

    def on_done(resp):
        assert resp.docs[0].matches[0].text == 'red'

    with f:
        f.search(inputs=doc, on_done=on_done)
