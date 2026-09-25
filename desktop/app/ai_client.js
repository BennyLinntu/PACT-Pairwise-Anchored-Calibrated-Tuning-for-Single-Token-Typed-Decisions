/* Talks to the local PACT-model backend (server.py, see tetris/README).
 * One HTTP call per piece: the engine works out every legal placement AND
 * what each one *does* to the board (lines cleared, holes, height,
 * bumpiness -- see engine.js:placementStats), so the model is never asked to
 * simulate Tetris physics from a raw board (it can't; it was never trained
 * to). It only has to read a short list of already-described outcomes and
 * pick the best one -- the same "typed decision over described choices"
 * shape PACT was actually fine-tuned for. All legality is still the
 * engine's job: every candidate sent is already a real legal placement, so
 * the model literally cannot pick an illegal move, only a bad one.
 *
 * Two-ply lookahead: a purely greedy, one-piece-at-a-time choice reliably
 * leaves a stray hole that blocks the very last cell of an otherwise
 * complete row, because nothing ever asks "does this choice still leave a
 * good move for the piece after this one?" So for every current-piece
 * placement we search all placements of the *known* next piece on top of
 * it and keep the best achievable combined outcome -- that combined score
 * is what ranks and (when capped) filters the candidates the model sees.
 * The model still makes the final call among them; the search only decides
 * what it's allowed to choose from. Cheap: a few hundred to ~1,000 grid
 * simulations, all plain array ops, comfortably under a game tick.
 *
 * The candidate list is capped at MAX_CANDIDATES (a schema field allows at
 * most 26 choices, one per letter code).
 *
 * Absolute URL, not a relative "/ai_move": this page can be loaded either
 * from server.py itself (http://127.0.0.1:8848/...) or from a packaged
 * desktop app (file://...), and a relative fetch would resolve against the
 * wrong origin in the second case. */

const AI_BACKEND = "http://127.0.0.1:8848";
// The schema technically allows up to 26 choices, and 16 fits the model's
// 2048-token prompt budget (26 does not: measured ~2800 tokens with the
// two-ply descriptions). But token budget isn't actually the binding
// constraint -- logged which index the model picked across real games and
// it wanders (2, 3, 8, 2, 0, 3, 3...), not reliably the heuristic's #1. With
// 16 shown, "not #1" can mean "#9", a real quality gap since the list is
// lookahead-score-sorted. Cutting to the top few means every option on
// offer is already near-optimal, so which one the model picks matters far
// less -- trading the model's freedom to choose for a guarantee that any
// choice is good, which is what "the model decides, but badly" was costing
// us in actual line clears.
const MAX_CANDIDATES = 6;

// Weights from the well-known Dellacherie/Lee four-feature Tetris heuristic
// (aggregate column-height sum, complete lines, holes, bumpiness) -- proven
// by genetic-algorithm tuning to rarely top out and to use the whole board
// width, not hand-tuned by us. Using *aggregate* height (sum over all
// columns) rather than the tallest column is what actually rewards
// spreading into empty columns instead of piling onto an already-tall one:
// building on a flat, empty column adds less to the sum than building on
// top of an existing stack of the same piece height would, whereas "tallest
// column" barely changes either way. Only used to choose which candidates
// the model gets to see -- the model still makes the final call.
//
// heightVariance is our own addition, not part of the original four. In
// play it still piled onto whichever side already had structure and left
// entire columns untouched: bumpiness only prices the *seam* between a
// tall region and an empty one, once, and nothing about that seam gets
// worse by building further from it -- so bumpiness alone never argues for
// crossing over to use the empty side. Variance over every column (not
// just neighbours) does: it's lower when height is spread across the board
// than concentrated in a few columns, regardless of adjacency. Weighted
// small relative to the proven terms above -- a tiebreaker for "which side
// to build on", not a reason to accept more holes or a taller stack.
function rankScore(stats) {
  return (
    -0.510066 * stats.aggregateHeight +
    0.760666 * stats.linesCleared +
    -0.35663 * stats.holes +
    -0.184483 * stats.bumpiness +
    -0.3 * stats.heightVariance
  );
}

window.checkAiBackend = async function (timeoutMs = 1500) {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(AI_BACKEND + "/health", { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return false;
    const data = await response.json();
    return !!data.ok;
  } catch (err) {
    return false;
  }
};

const TOPOUT_SCORE = -1e6; // a placement after which the next piece has nowhere legal to go

window.requestAiMove = async function (engine) {
  const snap = engine.snapshot();
  const currentType = snap.current.type;
  const nextType = snap.next[0];
  const placements1 = placementsFor(engine.grid, currentType);
  if (placements1.length === 0) return { rotation: engine.current.rotation, col: engine.current.col };

  const candidates = placements1.map((p1) => {
    const sim1 = simulatePlacement(engine.grid, currentType, p1);
    const placements2 = placementsFor(sim1.grid, nextType);
    let bestFollowUp = null;
    let lookaheadScore = TOPOUT_SCORE;
    for (const p2 of placements2) {
      const sim2 = simulatePlacement(sim1.grid, nextType, p2);
      const combined = { ...sim2, linesCleared: sim1.linesCleared + sim2.linesCleared };
      const score = rankScore(combined);
      if (score > lookaheadScore) { lookaheadScore = score; bestFollowUp = sim2; }
    }
    return { rotation: p1.rotation, col: p1.col, immediate: sim1, lookaheadScore, bestFollowUp };
  });

  candidates.sort((a, b) => b.lookaheadScore - a.lookaheadScore);

  // An immediate line clear is close to always correct -- rare enough that
  // it must never lose its slot to the MAX_CANDIDATES cap just because a
  // handful of taller-but-flatter non-clearing options score higher on the
  // general heuristic. Force every immediate-clear option into the list
  // (they're first anyway, sorted by lines cleared), then fill the rest by
  // lookahead score as before. The model still makes the actual choice.
  const clearers = candidates.filter((c) => c.immediate.linesCleared > 0)
    .sort((a, b) => b.immediate.linesCleared - a.immediate.linesCleared);
  const rest = candidates.filter((c) => c.immediate.linesCleared === 0);
  const top = [...clearers, ...rest].slice(0, MAX_CANDIDATES);

  const payload = {
    column_heights: snap.columnHeights,
    current: currentType,
    next: nextType,
    hold: snap.hold,
    lines: snap.lines,
    candidates: top.map((c) => ({
      rotation: c.rotation, col: c.col,
      lines_cleared: c.immediate.linesCleared, holes: c.immediate.holes,
      max_height: c.immediate.maxHeight, bumpiness: c.immediate.bumpiness,
      topout_next: c.bestFollowUp === null,
      followup_lines_cleared: c.bestFollowUp ? c.bestFollowUp.linesCleared : null,
      followup_holes: c.bestFollowUp ? c.bestFollowUp.holes : null,
      followup_max_height: c.bestFollowUp ? c.bestFollowUp.maxHeight : null,
      followup_bumpiness: c.bestFollowUp ? c.bestFollowUp.bumpiness : null,
    })),
  };

  const response = await fetch(AI_BACKEND + "/ai_move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error("AI backend error " + response.status);
  const data = await response.json();
  const chosen = top[data.index] ?? top[0];
  return { rotation: chosen.rotation, col: chosen.col };
};
