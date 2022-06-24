def getAlphabetMapping():
    return {num: chr(char) for (num, char) in zip(list(range(0, 26)), list(range(65, 65+26)))}


def readWordsFromFile(filepath):
    words = []

    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    return words
