from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np

# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenizeList(words: list[str]) -> list[str]:
    return [tokenizeAndLower(word) for word in words]


def tokenizeAndLower(sentence: str) -> list[str]:
    return [word.lower() for word in nltk.word_tokenize(sentence)]


def stem(words: list[str]):
    return [stemmer.stem(word) for word in words]


def bag_of_words(array_tokenized_words: list[str], all_words: list[str]):
    return_bag = np.zeros(len(all_words), dtype=np.float32)

    stem_words = stem(array_tokenized_words)

    for index, word in enumerate(all_words):
        if word in stem_words:
            return_bag[index] = 1

    return return_bag
