import re


NUMBER_WORDS = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}
STOP_WORDS = set("a an the of for to it its in on is are be with and that by".split())
NEGATIONS = {"no", "not", "never"}
WORD_PATTERN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_BOUNDARY.split(text) if sentence.strip()]


def content_words(text: str) -> tuple[str, ...]:
    return tuple(
        NUMBER_WORDS.get(word, word)
        for word in WORD_PATTERN.findall(text.casefold())
        if word not in STOP_WORDS
    )


def numbers(words: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(word for word in words if word[0].isdecimal())


def is_negated(words: tuple[str, ...]) -> bool:
    return not NEGATIONS.isdisjoint(words)


def token_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    total = len(left_set) + len(right_set)
    return 2 * len(left_set & right_set) / total if total else 0.0
