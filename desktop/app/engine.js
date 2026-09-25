/* Tetris engine - framework-free, rendering-agnostic.
 * Board is 10 columns x 20 visible rows (plus 2 hidden spawn rows above).
 * All state needed to render or to ask the model for a move is exposed via
 * engine.snapshot(). This file has no DOM/canvas code in it. */

const COLS = 10;
const ROWS = 20;
const HIDDEN_ROWS = 2; // pieces spawn here and become visible as they fall

const PIECES = {
  I: { color: "#4dd0e1", cells: [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]] },
  O: { color: "#ffe066", cells: [[1, 1], [1, 1]] },
  T: { color: "#c77dff", cells: [[0, 1, 0], [1, 1, 1], [0, 0, 0]] },
  S: { color: "#7bd88f", cells: [[0, 1, 1], [1, 1, 0], [0, 0, 0]] },
  Z: { color: "#ff6b6b", cells: [[1, 1, 0], [0, 1, 1], [0, 0, 0]] },
  J: { color: "#5c8df6", cells: [[1, 0, 0], [1, 1, 1], [0, 0, 0]] },
  L: { color: "#ffa94d", cells: [[0, 0, 1], [1, 1, 1], [0, 0, 0]] },
};
const PIECE_NAMES = Object.keys(PIECES);

function rotateMatrixCW(matrix) {
  const n = matrix.length;
  const out = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) out[c][n - 1 - r] = matrix[r][c];
  }
  return out;
}

function rotationsFor(cells) {
  const states = [cells];
  for (let i = 0; i < 3; i++) states.push(rotateMatrixCW(states[states.length - 1]));
  return states;
}

const ROTATIONS = {};
for (const name of PIECE_NAMES) ROTATIONS[name] = rotationsFor(PIECES[name].cells);

function pieceSpawnRow(type) { return type === "I" ? -1 : -2; }

/** Collision test against an arbitrary grid (not necessarily a live engine's)
 *  -- the primitive that lets the AI search hypothetical futures ("if I drop
 *  this piece here, where would the *next* piece legally land?") without a
 *  second TetrisEngine instance. */
function collidesOnGrid(grid, type, rotation, row, col) {
  const shape = ROTATIONS[type][rotation];
  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      if (!shape[r][c]) continue;
      const gr = row + r + HIDDEN_ROWS;
      const gc = col + c;
      if (gc < 0 || gc >= COLS || gr >= ROWS + HIDDEN_ROWS) return true;
      if (gr >= 0 && grid[gr][gc]) return true;
    }
  }
  return false;
}

/** Every legal (rotation, column) placement for `type` dropped onto `grid`,
 *  landing row included, deduplicated by resulting cell pattern. */
function placementsFor(grid, type) {
  const out = [];
  const seen = new Set();
  const spawnRow = pieceSpawnRow(type);
  const n = ROTATIONS[type].length;
  const rotationsToTry = type === "O" ? [0] : [...Array(n).keys()];
  for (const rotation of rotationsToTry) {
    const shape = ROTATIONS[type][rotation];
    const width = shape[0].length;
    let minC = 0, maxC = 0;
    for (let c = 0; c < width; c++) {
      if (shape.some((row) => row[c])) { minC = c; break; }
    }
    for (let c = width - 1; c >= 0; c--) {
      if (shape.some((row) => row[c])) { maxC = c; break; }
    }
    for (let col = -minC; col <= COLS - 1 - maxC; col++) {
      if (collidesOnGrid(grid, type, rotation, spawnRow, col)) continue;
      let row = spawnRow;
      while (!collidesOnGrid(grid, type, rotation, row + 1, col)) row++;
      const cells = [];
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (shape[r][c]) cells.push(`${row + r},${col + c}`);
        }
      }
      const key = cells.sort().join("|");
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ rotation, col, row });
    }
  }
  return out;
}

/** Simulate dropping `type` at `placement` onto `grid` (without mutating it)
 *  and report both the resulting grid and the standard Tetris heuristic
 *  features of it: lines cleared, holes, tallest column, aggregate height
 *  (sum over all columns) and bumpiness. The resulting grid is what lets a
 *  caller chain a second piece on top, for lookahead. */
function simulatePlacement(grid, type, placement) {
  const shape = ROTATIONS[type][placement.rotation];
  const next = grid.map((row) => row.slice());
  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      if (!shape[r][c]) continue;
      const gr = placement.row + r + HIDDEN_ROWS;
      const gc = placement.col + c;
      if (gr >= 0) next[gr][gc] = "X";
    }
  }
  let linesCleared = 0;
  const kept = [];
  for (const row of next) {
    if (row.every((cell) => cell)) linesCleared++;
    else kept.push(row);
  }
  while (kept.length < next.length) kept.unshift(new Array(COLS).fill(null));
  const visible = kept.slice(HIDDEN_ROWS);

  const heights = new Array(COLS).fill(0);
  for (let c = 0; c < COLS; c++) {
    for (let r = 0; r < visible.length; r++) {
      if (visible[r][c]) { heights[c] = visible.length - r; break; }
    }
  }
  let holes = 0;
  for (let c = 0; c < COLS; c++) {
    let seenBlock = false;
    for (let r = 0; r < visible.length; r++) {
      if (visible[r][c]) seenBlock = true;
      else if (seenBlock) holes++;
    }
  }
  let bumpiness = 0;
  for (let c = 0; c < COLS - 1; c++) bumpiness += Math.abs(heights[c] - heights[c + 1]);
  const aggregateHeight = heights.reduce((sum, h) => sum + h, 0);

  // Bumpiness only ever looks at *adjacent* columns, so it charges the cost
  // of a tall-region/empty-region boundary exactly once, at the seam --
  // nothing about placing more on the tall side or the empty side changes
  // that seam, so bumpiness alone gives no signal for "stop piling on one
  // side." Variance across *all* columns (not just neighbours) does: it's
  // lower when height is spread evenly across the board than when it's
  // concentrated in a few columns while others sit empty, regardless of
  // where the empty ones are relative to the tall ones.
  const meanHeight = aggregateHeight / COLS;
  const heightVariance = heights.reduce((sum, h) => sum + (h - meanHeight) ** 2, 0) / COLS;

  return { grid: kept, linesCleared, holes, maxHeight: Math.max(...heights), bumpiness,
           aggregateHeight, heightVariance };
}

function newBag(rng) {
  const bag = PIECE_NAMES.slice();
  for (let i = bag.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [bag[i], bag[j]] = [bag[j], bag[i]];
  }
  return bag;
}

function scoreForClear(lines, level) {
  const table = { 1: 100, 2: 300, 3: 500, 4: 800 };
  return (table[lines] || 0) * (level + 1);
}

class TetrisEngine {
  constructor(opts = {}) {
    this.rng = opts.rng || Math.random;
    this.onEvent = opts.onEvent || (() => {});
    this.reset();
  }

  reset() {
    this.grid = Array.from({ length: ROWS + HIDDEN_ROWS }, () => new Array(COLS).fill(null));
    this.bag = newBag(this.rng);
    this.nextQueue = [this.bag.shift(), this.bag.shift(), this.bag.shift()];
    this._refillBag();
    this.hold = null;
    this.holdUsed = false;
    this.score = 0;
    this.lines = 0;
    this.level = 0;
    this.gameOver = false;
    this.combo = -1;
    this._spawn();
    this.onEvent({ type: "reset" });
  }

  _refillBag() {
    while (this.nextQueue.length < 3) {
      if (this.bag.length === 0) this.bag = newBag(this.rng);
      this.nextQueue.push(this.bag.shift());
    }
  }

  _spawn() {
    const type = this.nextQueue.shift();
    this._refillBag();
    this.current = {
      type,
      rotation: 0,
      row: type === "I" ? -1 : -2,
      col: Math.floor(COLS / 2) - 2,
    };
    this.holdUsed = false;
    if (this._collides(this.current)) {
      this.gameOver = true;
      this.onEvent({ type: "gameover", score: this.score });
    }
  }

  _shape(piece) {
    return ROTATIONS[piece.type][piece.rotation];
  }

  _collides(piece) {
    const shape = this._shape(piece);
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (!shape[r][c]) continue;
        const gr = piece.row + r + HIDDEN_ROWS;
        const gc = piece.col + c;
        if (gc < 0 || gc >= COLS || gr >= ROWS + HIDDEN_ROWS) return true;
        if (gr >= 0 && this.grid[gr][gc]) return true;
      }
    }
    return false;
  }

  _move(dr, dc) {
    if (this.gameOver) return false;
    const moved = { ...this.current, row: this.current.row + dr, col: this.current.col + dc };
    if (this._collides(moved)) return false;
    this.current = moved;
    return true;
  }

  moveLeft() { if (this._move(0, -1)) this.onEvent({ type: "move" }); }
  moveRight() { if (this._move(0, 1)) this.onEvent({ type: "move" }); }

  softDrop() {
    if (this._move(1, 0)) {
      this.score += 1;
      this.onEvent({ type: "softdrop" });
      return true;
    }
    this._lock();
    return false;
  }

  hardDrop() {
    if (this.gameOver) return;
    let cells = 0;
    while (this._move(1, 0)) cells++;
    this.score += cells * 2;
    this._lock();
    this.onEvent({ type: "harddrop" });
  }

  rotateCW() { this._rotate(1); }
  rotateCCW() { this._rotate(-1); }

  _rotate(dir) {
    if (this.gameOver || this.current.type === "O") return;
    const n = ROTATIONS[this.current.type].length;
    const rotation = (this.current.rotation + dir + n) % n;
    const kicks = [0, -1, 1, -2, 2];
    for (const kick of kicks) {
      const candidate = { ...this.current, rotation, col: this.current.col + kick };
      if (!this._collides(candidate)) {
        this.current = candidate;
        this.onEvent({ type: "rotate" });
        return;
      }
    }
  }

  hold_() {
    if (this.gameOver || this.holdUsed) return;
    this.holdUsed = true;
    const type = this.current.type;
    if (this.hold === null) {
      this.hold = type;
      this._spawn();
    } else {
      const swapped = this.hold;
      this.hold = type;
      this.current = { type: swapped, rotation: 0, row: swapped === "I" ? -1 : -2,
                       col: Math.floor(COLS / 2) - 2 };
      if (this._collides(this.current)) {
        this.gameOver = true;
        this.onEvent({ type: "gameover", score: this.score });
      }
    }
    this.onEvent({ type: "hold" });
  }

  _lock() {
    const shape = this._shape(this.current);
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (!shape[r][c]) continue;
        const gr = this.current.row + r + HIDDEN_ROWS;
        const gc = this.current.col + c;
        if (gr >= 0) this.grid[gr][gc] = PIECES[this.current.type].color;
      }
    }
    const cleared = this._clearLines();
    if (cleared > 0) {
      this.combo += 1;
      this.lines += cleared;
      this.score += scoreForClear(cleared, this.level) + this.combo * 50;
      this.level = Math.floor(this.lines / 10);
      this.onEvent({ type: "lineclear", lines: cleared, score: this.score });
    } else {
      this.combo = -1;
    }
    if (!this.gameOver) this._spawn();
  }

  _clearLines() {
    let cleared = 0;
    for (let r = this.grid.length - 1; r >= 0; r--) {
      if (this.grid[r].every((cell) => cell)) {
        this.grid.splice(r, 1);
        this.grid.unshift(new Array(COLS).fill(null));
        cleared++;
        r++;
      }
    }
    return cleared;
  }

  gravityIntervalMs() {
    return Math.max(120, 800 - this.level * 60);
  }

  tick() {
    if (this.gameOver) return;
    if (!this._move(1, 0)) this._lock();
  }

  ghostRow() {
    let row = this.current.row;
    while (!this._collides({ ...this.current, row: row + 1 })) row++;
    return row;
  }

  /** Every valid (rotation, column) placement for the current piece, with the
   *  resulting landing row -- used both for a "drop preview" and to build the
   *  choice list the model picks from. */
  validPlacements() {
    return placementsFor(this.grid, this.current.type);
  }

  /** Simulate locking the piece at `placement` and report the standard
   *  Tetris heuristic features of the *resulting* board: lines cleared,
   *  holes (empty cells with a filled cell somewhere above, in the same
   *  column), the tallest column, aggregate height and bumpiness (roughness
   *  of the skyline). Used to describe each candidate placement's
   *  consequences to the model in plain language, instead of asking it to
   *  infer them from a raw board -- spatial simulation is the engine's job,
   *  not the model's. */
  placementStats(placement) {
    return simulatePlacement(this.grid, this.current.type, placement);
  }

  /** Column heights (0 = floor), 20 = full column, for a compact board summary. */
  columnHeights() {
    const heights = new Array(COLS).fill(0);
    for (let c = 0; c < COLS; c++) {
      for (let r = HIDDEN_ROWS; r < this.grid.length; r++) {
        if (this.grid[r][c]) { heights[c] = ROWS - (r - HIDDEN_ROWS); break; }
      }
    }
    return heights;
  }

  /** ASCII board, visible rows only, '.' empty, '#' filled, current piece as 'O'. */
  asciiBoard() {
    const shape = this._shape(this.current);
    const overlay = this.grid.slice(HIDDEN_ROWS).map((row) => row.slice());
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (!shape[r][c]) continue;
        const gr = this.current.row + r;
        const gc = this.current.col + c;
        if (gr >= 0 && gr < ROWS && gc >= 0 && gc < COLS) overlay[gr][gc] = "X";
      }
    }
    return overlay.map((row) => row.map((cell) => (cell ? (cell === "X" ? "X" : "#") : ".")).join("")).join("\n");
  }

  snapshot() {
    return {
      grid: this.grid.slice(HIDDEN_ROWS),
      current: { ...this.current },
      currentShape: this._shape(this.current),
      hold: this.hold,
      next: this.nextQueue.slice(),
      score: this.score,
      lines: this.lines,
      level: this.level,
      gameOver: this.gameOver,
      ghostRow: this.gameOver ? null : this.ghostRow(),
      columnHeights: this.columnHeights(),
      asciiBoard: this.asciiBoard(),
    };
  }
}

if (typeof module !== "undefined") module.exports = { TetrisEngine, PIECES, ROTATIONS, COLS, ROWS, HIDDEN_ROWS };
