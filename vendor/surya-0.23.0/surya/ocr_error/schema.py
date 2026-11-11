from typing import List

from pydantic import BaseModel


class OCRErrorDetectionResult(BaseModel):
    texts: List[str]
    labels: List[str]
    # P(bad) per text (softmax of the "bad" class). Lets callers gate expensive
    # follow-up work on confidence instead of just the argmax label.
    scores: List[float] = []
