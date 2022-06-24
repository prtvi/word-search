DIRECTIONS = {
    1: (-1, -1), 2: (0, -1),  3: (1, -1),
    4: (-1, 0), 6: (1, 0),
    7: (-1, 1),  8: (0, 1), 9: (1, 1)
}


class Word:

    def __init__(self, word: str) -> None:
        self.word = word.upper()
        self.len = len(self.word)

        self.rows = None
        self.cols = None

        self.direction = None

    def search(self, row: int, col: int, grid: list) -> bool:

        if grid[row][col] != self.word[0]:
            return False, False

        for direction, (y, x) in DIRECTIONS.items():

            rd, cd = row + x, col + y
            flag = True

            for k in range(1, len(self.word)):

                if 0 <= rd < self.rows and 0 <= cd < self.cols and self.word[k] == grid[rd][cd]:
                    rd += x
                    cd += y
                else:
                    flag = False
                    break

            if flag:
                self.direction = direction
                return True, self.direction

        return False, False

    def pattern_search(self, grid: list) -> None:

        self.rows = len(grid)
        self.cols = len(grid[0])

        for row in range(self.rows):
            for col in range(self.cols):
                found, dir = self.search(row, col, grid)
                if found:
                    end_col = col + DIRECTIONS[dir][0] * (self.len - 1)
                    end_row = row + DIRECTIONS[dir][1] * (self.len - 1)

                    # print(
                    #     f"{self.word}: ({col}, {row}) -> ({end_col}, {end_row})")

                    return (self.word, (col, row), (end_col, end_row))


class WordSearch:

    def __init__(self, wordsArr: list, grid: list) -> None:
        self.words = wordsArr
        self.nWords = len(self.words)

        self.grid = grid

        self.wordsFound = []
        self.nWordsFound = 0

        self.summary = {}

    def findAll(self) -> list:

        for word in self.words:
            found = Word(word).pattern_search(self.grid)
            if found:
                self.wordsFound.append(found)

        self.nWordsFound = len(self.wordsFound)

        self.summary = {
            'nWordsInput': self.nWords,
            'nWordsFound': self.nWordsFound
        }

        return self.wordsFound, self.summary
