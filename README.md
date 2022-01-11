# TransformerQAExecutor

**TransformerQAExecutor** wraps a question-answering model from Hugging Face. Given a question and paragraph/context, it extracts the relevant answers from the given paragraph/context.

**TransformerQAExexecutor** receives [`Document`s](https://docs.jina.ai/fundamentals/document/) where each `Document`'s `text` is the paragraph, `doc.tags['question']` stores the question.

## Usage

Use the prebuilt images from Jina Hub in your Flow:

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TransformerQAExecutor')

doc = Document(
    text='Apple is a pome fruit that is red in color. Banana is yellow.',
    tags={'question': 'what is the color of an apple?'},
)

with f:
    doc = f.search(inputs=doc, return_results=True).docs[0]
```

After searching, we can perform the following to get the top 1 answer.
```python
# Top 1 match
match = doc.matches[0]
answer = match.text
confidence = match.scores['confidence'].value
```
We can also get the confidence score of the answer.


### Use other pre-trained models
By default, the `TransformerQAExecutor` uses the `bert-large-uncased-whole-word-masking-finetuned-squad`.

You can use other models by specifying the `model_name` parameter:

```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://TransformerTorchEncoder',
    uses_with={'model_name': 'distilbert-base-cased-distilled-squad'}
)
```
You can check the supported pre-trained models [here](https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads).

By default, the name of the tokenizer that `TransformerQAExecutor` uses will be the same the model provided. 

You may use your own tokenizer by specifying the `tokenizer_name` parameter:

```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://TransformerTorchEncoder',
    uses_with={'tokenizer_name': 'distilbert-base-cased-distilled-squad'}
)
```

### Use GPUs
To enable GPU, you can set the `device` parameter to a CUDA device.
Make sure your machine is CUDA-compatible and that you've installed all the required libraries and drivers.
If you're using a docker container, make sure to add the `gpu` tag and enable 
GPU access to Docker with `gpus='all'`.
Furthermore, make sure you satisfy the prerequisites mentioned in 
[Executor on GPU tutorial](https://docs.jina.ai/tutorials/gpu_executor/#prerequisites).

```python

from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://TransformerQAExecutor',
    uses_with={'device': 'cuda'}, gpus='all'
)
```

## Reference
- [Huggingface Question-Answering](https://huggingface.co/models?pipeline_tag=question-answering&sort=downloads)
