from enum import IntEnum
from typing import List, Optional

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
        **kwargs
    ):
        tokenizer_name = tokenizer_name or model_name
        super().__init__(**kwargs)
        self.model = pipeline(
            'question-answering',
            model=model_name,
            tokenizer=tokenizer_name,
            device=Device[device],
        )

    def _concatenate_matches(self, matches: DocumentArray):
        """
        Helper function that to concatenate matches and
          keep track of the match positions within the
          concatenated string
        """
        concatenated_matches = matches[0].text + ' '
        span_ranges_to_matches = [(0, len(concatenated_matches), 0)]
        for i, match in enumerate(matches[1:], start=1):

            padded_match_text = match.text + ' '
            _, prev_max, _ = span_ranges_to_matches[i - 1]
            current_min = prev_max
            current_max = current_min + len(padded_match_text)
            # Collect span beginning and end as well as match id
            span_ranges_to_matches.append((current_min, current_max, i))
            concatenated_matches += padded_match_text

        return concatenated_matches, span_ranges_to_matches

    def _filter_matches(
        self,
        qa_outputs: List[dict],
        span_ranges_to_matches: List[tuple],
        all_matches: DocumentArray,
    ) -> DocumentArray:
        """Get a match for each of the qa outputs"""
        ordered_matches = []
        for qa_output in qa_outputs:
            span_start_pos = qa_output['start']
            # Go through all span ranges for all matches
            for span_range_to_match in span_ranges_to_matches:
                span_range_min, span_range_max, match_index = span_range_to_match
                # Check only based on span starting pos to avoid
                # case of span spanning two matchesspan spanning two matches
                above_span_min = span_range_min <= span_start_pos
                below_span_max = span_start_pos < span_range_max
                # Find match that containing text that covers span
                if above_span_min and below_span_max:
                    # Collect the match
                    ordered_matches.append(all_matches[match_index])
                    # Stop inner loop and go to next qa output
                    break
        return ordered_matches

    @requests(on='/search')
    def generate(self, docs, **kwargs):
        for doc in docs:
            concatenated_texts, span_ranges_to_matches = self._concatenate_matches(
                doc.matches
            )
            qa_input = [{'context': concatenated_texts, 'question': doc.text}]
            qa_outputs = self.model(qa_input, top_k=10)

            # Get matches in order of qa outputs
            ordered_matches = self._filter_matches(
                qa_outputs, span_ranges_to_matches, doc.matches
            )

            # Every relevant output is matched to match it was extracted from
            assert len(qa_outputs) == len(ordered_matches)

            targets = [o['answer'] for o in qa_outputs]
            confidence_scores = [o['score'] for o in qa_outputs]
            uris = [m.uri for m in ordered_matches]
            tags = [m.tags for m in ordered_matches]
            paragraphs = [m.text for m in ordered_matches]

            doc.matches.clear()
            for (target, confidence_score, uri, tag, paragraph,) in zip(
                targets,
                confidence_scores,
                uris,
                tags,
                paragraphs,
            ):
                match = Document(
                    text=target,
                    scores={
                        'confidence': confidence_score,
                    },
                    uri=uri,
                    tags=tag,
                )
                match.tags['paragraph'] = paragraph
                doc.matches.append(match)

            doc.matches.sort(lambda m: m.scores['confidence'].value, reverse=True)
