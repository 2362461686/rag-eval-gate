"""Sentence-level groundedness scoring; the entailment model is injected."""
import re

_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[a-z0-9]+")
_NON_CLAIM = re.compile(r"^(sure|certainly|of course|here is|here are|i hope|thank you)\b", re.IGNORECASE)
MIN_CLAIM_WORDS = 4


def words(text):
    return _WORD.findall(text.lower())


def split_sentences(text):
    """Return (start, end, sentence) triples with offsets into `text`."""
    spans = []
    for line in re.finditer(r"[^\n]+", text):
        offset = line.start()
        cursor = 0
        body = line.group()
        for boundary in _SENTENCE_END.finditer(body):
            spans.append((offset + cursor, offset + boundary.start(), body[cursor : boundary.start()]))
            cursor = boundary.end()
        spans.append((offset + cursor, offset + len(body), body[cursor:]))
    return [(start, end, sentence.strip()) for start, end, sentence in spans if sentence.strip()]


def is_claim(sentence):
    """Skip boilerplate and fragments that assert nothing checkable."""
    if sentence.endswith(":") or _NON_CLAIM.match(sentence):
        return False
    return len(words(sentence)) >= MIN_CLAIM_WORDS


def evidence_windows(evidence, size=60, stride=30):
    """Overlapping word windows that together cover the whole evidence."""
    tokens = evidence.split()
    if not tokens:
        return []
    last = max(len(tokens) - size, 0)
    return [" ".join(tokens[start : start + size]) for start in [*range(0, last, stride), last]]


def overlap(sentence, window):
    """Share of the sentence's distinct words that occur in the window."""
    claim = set(words(sentence))
    return len(claim & set(words(window))) / len(claim) if claim else 0.0


def nearest_windows(sentence, windows, top=3):
    order = sorted(range(len(windows)), key=lambda i: -overlap(sentence, windows[i]))
    return [windows[i] for i in order[:top]]


def sentence_supports(sentences, evidence, entail, top=3):
    """Best entailment probability of each sentence over its nearest evidence windows."""
    windows = evidence_windows(evidence)
    if not sentences:
        return []
    if not windows:
        return [0.0 for _ in sentences]
    pairs, owners = [], []
    for position, sentence in enumerate(sentences):
        for window in nearest_windows(sentence, windows, top):
            pairs.append((window, sentence))
            owners.append(position)
    probabilities = entail(pairs)
    best = [0.0] * len(sentences)
    for position, probability in zip(owners, probabilities):
        best[position] = max(best[position], float(probability))
    return best


def lexical_supports(sentences, evidence):
    """Baseline: the share of each sentence's words found anywhere in the evidence."""
    return [overlap(sentence, evidence) for sentence in sentences]


def hallucination_score(supports):
    """A response is as suspect as its least supported claim."""
    return 1.0 - min(supports) if supports else 0.0


def span_overlaps(start, end, labels):
    return any(label["start"] < end and start < label["end"] for label in labels)
