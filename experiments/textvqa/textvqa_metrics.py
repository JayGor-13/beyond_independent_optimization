"""TextVQA/VQA answer normalization and soft-accuracy scoring."""

from __future__ import annotations

import re
from collections import Counter


NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "maam": "ma'am", "mightnt": "mightn't", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't",
    "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyll": "somebody'll", "somebodys": "somebody's",
    "someonell": "someone'll", "someones": "someone's", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
PUNCTUATION = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def extract_answer(text: str) -> str:
    """Remove common generation wrappers without truncating multi-word answers."""
    line = next((line.strip() for line in str(text).splitlines() if line.strip()), "")
    line = re.sub(r"^(?:final\s+)?answer\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^the answer is\s+", "", line, flags=re.IGNORECASE)
    return line.strip().strip('"\'')


def normalize_answer(answer: str) -> str:
    text = extract_answer(answer).lower().replace("\n", " ").replace("\t", " ").strip()
    for mark in PUNCTUATION:
        if (mark + " " in text or " " + mark in text) or COMMA_STRIP.search(text):
            text = text.replace(mark, "")
        else:
            text = text.replace(mark, " ")
    text = PERIOD_STRIP.sub("", text)
    words = []
    for word in text.split():
        word = NUMBER_MAP.get(word, word)
        if word not in ARTICLES:
            words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_soft_score(prediction: str, answers: list[str]) -> float:
    """Official VQA consensus score using the ten human reference answers.

    For each annotator answer, agreement is computed against the other nine
    annotators and capped at one after three matches, then averaged.
    """
    predicted = normalize_answer(prediction)
    normalized = [normalize_answer(answer) for answer in answers]
    if not predicted or not normalized:
        return 0.0
    matches = Counter(normalized)[predicted]
    total = len(normalized)
    scores = []
    for answer in normalized:
        other_matches = matches - int(answer == predicted)
        scores.append(min(1.0, other_matches / 3.0))
    return sum(scores) / total

