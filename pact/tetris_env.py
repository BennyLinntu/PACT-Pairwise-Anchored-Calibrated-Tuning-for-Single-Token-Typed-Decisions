"""Pure-Python port of desktop/app/engine.js's simulation core, for
generating Tetris RL training states offline without a browser or a JS
bridge. Mirrors piece shapes, rotation, collision, placement search, the
two-ply lookahead stats, and the heightVariance-augmented Dellacherie/Lee
ranking exactly, so states line up with what live play would show.
Rendering, hold, and pause are not needed here and are left out.
"""
import random

COLS = 10
ROWS = 20
HIDDEN_ROWS = 2

PIECES = {
    "I": [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
    "S": [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
    "Z": [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
    "J": [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
    "L": [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
}
PIECE_NAMES = list(PIECES)


def _rotate_cw(matrix):
    n = len(matrix)
    return [[matrix[n - 1 - c][r] for c in range(n)] for r in range(n)]


def _rotations_for(cells):
    states = [cells]
    for _ in range(3):
        states.append(_rotate_cw(states[-1]))
    return states


ROTATIONS = {name: _rotations_for(cells) for name, cells in PIECES.items()}


def spawn_row(piece_type):
    return -1 if piece_type == "I" else -2


def collides(grid, piece_type, rotation, row, col):
    shape = ROTATIONS[piece_type][rotation]
    for r, row_cells in enumerate(shape):
        for c, v in enumerate(row_cells):
            if not v:
                continue
            gr = row + r + HIDDEN_ROWS
            gc = col + c
            if gc < 0 or gc >= COLS or gr >= ROWS + HIDDEN_ROWS:
                return True
            if gr >= 0 and grid[gr][gc]:
                return True
    return False


def placements_for(grid, piece_type):
    """Every legal (rotation, column) placement, deduped by resulting cells --
    mirrors engine.js:placementsFor exactly."""
    out = []
    seen = set()
    n = len(ROTATIONS[piece_type])
    rotations_to_try = [0] if piece_type == "O" else list(range(n))
    for rotation in rotations_to_try:
        shape = ROTATIONS[piece_type][rotation]
        width = len(shape[0])
        cols_with = [c for c in range(width) if any(row[c] for row in shape)]
        min_c, max_c = min(cols_with), max(cols_with)
        for col in range(-min_c, COLS - max_c):
            srow = spawn_row(piece_type)
            if collides(grid, piece_type, rotation, srow, col):
                continue
            row = srow
            while not collides(grid, piece_type, rotation, row + 1, col):
                row += 1
            cells = []
            for r, row_cells in enumerate(shape):
                for c, v in enumerate(row_cells):
                    if v:
                        cells.append((row + r, col + c))
            key = tuple(sorted(cells))
            if key in seen:
                continue
            seen.add(key)
            out.append({"rotation": rotation, "col": col, "row": row})
    return out


def simulate_placement(grid, piece_type, placement):
    """Drop `piece_type` at `placement` onto `grid` (not mutated) and report
    the resulting grid plus the standard heuristic features -- mirrors
    engine.js:simulatePlacement exactly, including heightVariance."""
    shape = ROTATIONS[piece_type][placement["rotation"]]
    next_grid = [row[:] for row in grid]
    for r, row_cells in enumerate(shape):
        for c, v in enumerate(row_cells):
            if not v:
                continue
            gr = placement["row"] + r + HIDDEN_ROWS
            gc = placement["col"] + c
            if gr >= 0:
                next_grid[gr][gc] = 1
    lines_cleared = 0
    kept = []
    for row in next_grid:
        if all(row):
            lines_cleared += 1
        else:
            kept.append(row)
    while len(kept) < len(next_grid):
        kept.insert(0, [0] * COLS)
    visible = kept[HIDDEN_ROWS:]

    heights = [0] * COLS
    for c in range(COLS):
        for r in range(len(visible)):
            if visible[r][c]:
                heights[c] = len(visible) - r
                break
    holes = 0
    for c in range(COLS):
        seen_block = False
        for r in range(len(visible)):
            if visible[r][c]:
                seen_block = True
            elif seen_block:
                holes += 1
    bumpiness = sum(abs(heights[c] - heights[c + 1]) for c in range(COLS - 1))
    aggregate_height = sum(heights)
    mean_height = aggregate_height / COLS
    height_variance = sum((h - mean_height) ** 2 for h in heights) / COLS

    return {"grid": kept, "linesCleared": lines_cleared, "holes": holes,
            "maxHeight": max(heights), "bumpiness": bumpiness,
            "aggregateHeight": aggregate_height, "heightVariance": height_variance}


def rank_score(stats):
    """Mirrors ai_client.js:rankScore exactly (Dellacherie/Lee + our
    heightVariance balance term)."""
    return (
        -0.510066 * stats["aggregateHeight"]
        + 0.760666 * stats["linesCleared"]
        + -0.35663 * stats["holes"]
        + -0.184483 * stats["bumpiness"]
        + -0.3 * stats["heightVariance"]
    )


def column_heights(grid):
    heights = [0] * COLS
    for c in range(COLS):
        for r in range(HIDDEN_ROWS, len(grid)):
            if grid[r][c]:
                heights[c] = ROWS - (r - HIDDEN_ROWS)
                break
    return heights


class TetrisGame:
    """Minimal headless game: reset/apply, 7-bag -- matches engine.js's
    TetrisEngine for the subset RL training needs (no hold, no pause, no
    rendering)."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self.grid = [[0] * COLS for _ in range(ROWS + HIDDEN_ROWS)]
        self.bag = self._new_bag()
        self.next_queue = [self.bag.pop(0), self.bag.pop(0), self.bag.pop(0)]
        self._refill()
        self.score = 0
        self.lines = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current = self.next_queue.pop(0)
        self._refill()

    def _new_bag(self):
        bag = PIECE_NAMES[:]
        self.rng.shuffle(bag)
        return bag

    def _refill(self):
        while len(self.next_queue) < 3:
            if not self.bag:
                self.bag = self._new_bag()
            self.next_queue.append(self.bag.pop(0))

    def apply(self, placement):
        """Lock `placement` (one entry from placements_for(self.grid,
        self.current)), clear lines, score, and advance to the next piece."""
        result = simulate_placement(self.grid, self.current, placement)
        self.grid = result["grid"]
        if result["linesCleared"]:
            table = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += table.get(result["linesCleared"], 0)
            self.lines += result["linesCleared"]
        self.pieces_placed += 1
        self.current = self.next_queue.pop(0)
        self._refill()
        if not placements_for(self.grid, self.current):
            self.game_over = True
        return result
