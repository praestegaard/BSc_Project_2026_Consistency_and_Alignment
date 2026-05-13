import Levenshtein

# Levenshtein Ratio
# https://www.geeksforgeeks.org/python/introduction-to-python-levenshtein-module/#pythonlevenshtein-module
def levenshtein_ratio(response_1 : str, response_2 : str) -> float:
    ratio = Levenshtein.ratio(response_1, response_2)

    return ratio * 100