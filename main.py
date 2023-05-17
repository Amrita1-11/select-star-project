import re
import json
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from flask import Flask, jsonify

app = Flask(__name__)

stop_words = set(stopwords.words("english"))


def normalize_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\- ]", "", text.lower())
    words = word_tokenize(text)

    # Filter out stop words
    words = [word for word in words if word not in stop_words]

    # Handle hyphenated words
    hyphenated_words = []
    for word in words:
        if "-" in word:
            hyphenated_words.extend(word.split("-"))
        else:
            hyphenated_words.append(word)

    # Stemming using Porter's algorithm
    stemmer = nltk.PorterStemmer()
    normalized_words = [stemmer.stem(word) for word in hyphenated_words]

    return normalized_words

def build_inverted_index(file_path):
    inverted_index = defaultdict(list)

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines, start=1):
            words = normalize_text(line)
            for word in words:
                inverted_index[word].append(line_number)

    return inverted_index


@app.route("/index/<word>")
def search_word(word):
    # Normalize the search word
    normalized_word = normalize_text(word)[0]

    # Search for the word in the inverted index
    if normalized_word in inverted_index:
        count = len(inverted_index[normalized_word])
        lines = inverted_index[normalized_word]
        response = {"count": count, "lines": lines}
    else:
        response = {"count": 0, "lines": []}

    return jsonify(response)


if __name__ == "__main__":
    file_path = "example.txt"  # Replace with the actual file path
    inverted_index = build_inverted_index(file_path)
    print(inverted_index)
    app.run()
