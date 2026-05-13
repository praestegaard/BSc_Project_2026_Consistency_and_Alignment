import difflib

# Sequence Matcher
# Inspired by https://www.codespeedy.com/sequencematcher-in-python/
def sequence_matcher(response_1 : str, response_2 : str) -> float:
    ratio = difflib.SequenceMatcher(None, response_1, response_2).ratio()
    return ratio * 100.0