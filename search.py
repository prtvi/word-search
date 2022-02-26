
grid = [
    # 0    1    2    3    4    5    6    7    8    9
    ['C', 'L', 'A', 'S', 'S', 'E', 'S', 'M', 'O', 'P'],  # 0
    ['H', 'T', 'I', 'D', 'M', 'I', 'Y', 'U', 'I', 'E'],  # 1
    ['H', 'E', 'B', 'Z', 'B', 'E', 'P', 'T', 'T', 'L'],  # 2
    ['E', 'S', 'R', 'L', 'M', 'M', 'P', 'A', 'E', 'B'],  # 3
    ['A', 'A', 'A', 'E', 'O', 'D', 'A', 'B', 'R', 'I'],  # 4
    ['R', 'F', 'C', 'H', 'D', 'C', 'H', 'L', 'A', 'X'],  # 5
    ['R', 'U', 'O', 'H', 'U', 'O', 'K', 'E', 'T', 'E'],  # 6
    ['A', 'R', 'B', 'H', 'L', 'A', 'C', 'S', 'O', 'L'],  # 7
    ['Y', 'P', 'P', 'Y', 'E', 'I', 'N', 'S', 'R', 'F'],  # 8
    ['E', 'S', 'S', 'T', 'C', 'E', 'J', 'B', 'O', 'S']   # 9
]

words = ['ruby', 'blocks', 'heredocs', 'classes', 'iterator', 'module',
         'objects', 'flexible', 'each', 'happy', 'mutable', 'lambda', 'hash', 'array']


class word:

    def __init__(self, word: str) -> None:

        self.word = word.upper()
        self.len = len(self.word)

        self.rows = None
        self.cols = None

        self.directions = {
            1: (-1, -1), 2: (0, -1),  3: (1, -1),
            4: (-1, 0), 6: (1, 0),
            7: (-1, 1),  8: (0, 1), 9: (1, 1)
        }
        self.dir = None

    def search(self, row: int, col: int, grid: list) -> None:

        if grid[row][col] != self.word[0]:
            return False, False

        for dir, (y, x) in self.directions.items():

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
                self.dir = dir
                return True, self.dir

        return False, False

    def pattern_search(self, grid: list) -> None:

        self.rows = len(grid)
        self.cols = len(grid[0])

        for row in range(self.rows):
            for col in range(self.cols):
                found, dir = self.search(row, col, grid)
                if found:
                    print(f"{self.word}: ({col}, {row})")


for w in words:
    word(w).pattern_search(grid)
