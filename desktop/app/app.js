/* Rendering, input and game-loop glue. Reads the two-letter language pack
 * from window.LANG (set inline by index_en.html / index_zh.html) so the
 * engine and this file are identical between the two language versions. */

(function () {
  const CELL = 28;
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  const holdCanvas = document.getElementById("hold-canvas");
  const holdCtx = holdCanvas.getContext("2d");
  const nextCanvases = [0, 1, 2].map((i) => document.getElementById("next-canvas-" + i));

  canvas.width = engineCols() * CELL;
  canvas.height = engineRows() * CELL;

  function engineCols() { return 10; }
  function engineRows() { return 20; }

  let engine;
  engine = new TetrisEngine({ onEvent: onEngineEvent });
  let aiMode = false;
  let aiBusy = false;
  let aiTimer = null;
  let paused = false;
  let gravityTimer = null;

  const L = window.LANG;

  function drawCell(context, r, c, color, size) {
    const x = c * size, y = r * size;
    context.fillStyle = color;
    context.fillRect(x, y, size, size);
    context.strokeStyle = "rgba(0,0,0,0.25)";
    context.lineWidth = 1;
    context.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
    context.fillStyle = "rgba(255,255,255,0.18)";
    context.fillRect(x, y, size, 3);
  }

  function draw() {
    if (!engine) return;
    const snap = engine.snapshot();
    ctx.fillStyle = "#12151c";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    for (let c = 1; c < engineCols(); c++) {
      ctx.beginPath(); ctx.moveTo(c * CELL, 0); ctx.lineTo(c * CELL, canvas.height); ctx.stroke();
    }
    for (let r = 1; r < engineRows(); r++) {
      ctx.beginPath(); ctx.moveTo(0, r * CELL); ctx.lineTo(canvas.width, r * CELL); ctx.stroke();
    }

    // settled cells
    for (let r = 0; r < snap.grid.length; r++) {
      for (let c = 0; c < snap.grid[r].length; c++) {
        if (snap.grid[r][c]) drawCell(ctx, r, c, snap.grid[r][c], CELL);
      }
    }

    // ghost
    if (snap.ghostRow !== null) {
      const shape = snap.currentShape;
      ctx.fillStyle = "rgba(255,255,255,0.10)";
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (!shape[r][c]) continue;
          const gr = snap.ghostRow + r, gc = snap.current.col + c;
          if (gr >= 0 && gr < engineRows()) ctx.fillRect(gc * CELL, gr * CELL, CELL, CELL);
        }
      }
    }

    // current piece
    if (!snap.gameOver) {
      const shape = snap.currentShape;
      const color = enginePieceColor(snap.current.type);
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (!shape[r][c]) continue;
          const gr = snap.current.row + r, gc = snap.current.col + c;
          if (gr >= 0 && gr < engineRows()) drawCell(ctx, gr, gc, color, CELL);
        }
      }
    }

    if (snap.gameOver) {
      ctx.fillStyle = "rgba(10,10,14,0.72)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 22px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(L.gameOver, canvas.width / 2, canvas.height / 2 - 10);
      ctx.font = "14px system-ui, sans-serif";
      ctx.fillText(L.pressNewGame, canvas.width / 2, canvas.height / 2 + 16);
    }

    document.getElementById("score").textContent = snap.score;
    document.getElementById("lines").textContent = snap.lines;
    document.getElementById("level").textContent = snap.level;

    drawMini(holdCtx, holdCanvas, snap.hold ? [snap.hold] : []);
    snap.next.forEach((type, i) => drawMini(nextCanvases[i].getContext("2d"), nextCanvases[i], [type]));
  }

  function enginePieceColor(type) {
    const colors = { I: "#4dd0e1", O: "#ffe066", T: "#c77dff", S: "#7bd88f",
                     Z: "#ff6b6b", J: "#5c8df6", L: "#ffa94d" };
    return colors[type];
  }

  function drawMini(context, cv, types) {
    context.fillStyle = "#12151c";
    context.fillRect(0, 0, cv.width, cv.height);
    if (!types.length) return;
    const type = types[0];
    const shape = window.ROTATIONS_FOR(type);
    const size = 18;
    const rows = shape.length, cols = shape[0].length;
    const offX = (cv.width - cols * size) / 2, offY = (cv.height - rows * size) / 2;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (!shape[r][c]) continue;
        context.fillStyle = enginePieceColor(type);
        context.fillRect(offX + c * size, offY + r * size, size, size);
        context.strokeStyle = "rgba(0,0,0,0.25)";
        context.strokeRect(offX + c * size + 0.5, offY + r * size + 0.5, size - 1, size - 1);
      }
    }
  }

  const session = { games: 0, bestScore: 0, totalLines: 0 };
  window.pactSession = session; // queryable from outside (devtools, screenshots-by-JS, etc.)

  function onEngineEvent(evt) {
    if (evt.type === "gameover") {
      stopGravity();
      session.games += 1;
      session.bestScore = Math.max(session.bestScore, engine.score);
      session.totalLines += engine.lines;
      if (aiMode) {
        // Keep playing: a finished game is not a reason to sit idle when the
        // whole point was "let it play" -- start the next one after a beat
        // so the final board is visible for a moment first.
        document.getElementById("ai-status").textContent =
          `${L.gameOver} (#${session.games}, best ${session.bestScore}) -- ${L.aiWatching}`;
        setTimeout(() => { if (aiMode) newGame(true); }, 1500);
      } else {
        setAiMode(false);
      }
    }
    draw();
  }

  function startGravity() {
    stopGravity();
    const step = () => {
      engine.tick();
      draw();
      if (!engine.gameOver) gravityTimer = setTimeout(step, engine.gravityIntervalMs());
    };
    gravityTimer = setTimeout(step, engine.gravityIntervalMs());
  }
  function stopGravity() { if (gravityTimer) clearTimeout(gravityTimer); gravityTimer = null; }

  function newGame(keepAiMode = false) {
    engine.reset();
    paused = false;
    if (keepAiMode && aiMode) {
      // aiMode is already true; just (re)kick the move loop for the fresh game.
      scheduleAiMove();
    } else {
      setAiMode(false);
    }
    startGravity();
    draw();
  }

  function togglePause() {
    if (engine.gameOver) return;
    paused = !paused;
    document.getElementById("pause-btn").textContent = paused ? L.resume : L.pause;
    if (paused) { stopGravity(); if (aiTimer) clearTimeout(aiTimer); }
    else { startGravity(); if (aiMode) scheduleAiMove(); }
  }

  document.addEventListener("keydown", (e) => {
    if (engine.gameOver || paused || aiMode) return;
    if (e.code === "Space") { e.preventDefault(); engine.hardDrop(); draw(); return; }
    switch (e.key) {
      case "ArrowLeft": engine.moveLeft(); draw(); break;
      case "ArrowRight": engine.moveRight(); draw(); break;
      case "ArrowDown": engine.softDrop(); draw(); break;
      case "ArrowUp": engine.rotateCW(); draw(); break;
      case "z": case "Z": engine.rotateCCW(); draw(); break;
      case " ": case "Spacebar": e.preventDefault(); engine.hardDrop(); draw(); break;
      case "c": case "C": case "Shift": engine.hold_(); draw(); break;
      case "p": case "P": togglePause(); break;
      default: return;
    }
  });

  // ---- AI play ----------------------------------------------------------

  function setAiMode(on) {
    aiMode = on;
    document.getElementById("ai-btn").textContent = aiMode ? L.aiStop : L.aiStart;
    document.getElementById("ai-status").textContent = aiMode ? L.aiWatching : "";
    document.getElementById("ai-btn").classList.toggle("active", aiMode);
    if (aiTimer) { clearTimeout(aiTimer); aiTimer = null; }
    if (aiMode && !engine.gameOver && !paused) scheduleAiMove();
  }

  function scheduleAiMove(delay = 250) {
    if (aiTimer) clearTimeout(aiTimer);
    aiTimer = setTimeout(runAiMove, delay);
  }

  async function runAiMove() {
    if (!aiMode || engine.gameOver || paused || aiBusy) return;
    aiBusy = true;
    document.getElementById("ai-status").textContent = L.aiThinking;
    try {
      const plan = await window.requestAiMove(engine);
      if (aiMode && !engine.gameOver) applyAiPlan(plan);
    } catch (err) {
      document.getElementById("ai-status").textContent = L.aiError;
      console.error(err);
    } finally {
      aiBusy = false;
      if (aiMode && !engine.gameOver) {
        document.getElementById("ai-status").textContent = L.aiWatching;
        scheduleAiMove(350);
      }
    }
  }

  function applyAiPlan(plan) {
    // plan: {rotation: 0-3, col: target column}. Rotate then walk to the column,
    // then hard drop -- executed as discrete steps so it's visible on screen.
    let guard = 0;
    while (engine.current.rotation !== plan.rotation && guard++ < 4) engine.rotateCW();
    while (engine.current.col < plan.col && guard++ < 20) engine.moveRight();
    while (engine.current.col > plan.col && guard++ < 20) engine.moveLeft();
    engine.hardDrop();
    draw();
  }

  document.getElementById("new-game-btn").addEventListener("click", () => newGame(false));
  document.getElementById("pause-btn").addEventListener("click", togglePause);
  document.getElementById("ai-btn").addEventListener("click", () => {
    if (document.getElementById("ai-btn").disabled) return;
    setAiMode(!aiMode);
  });

  // ---- AI backend availability -------------------------------------------
  // The packaged app ships manual play only; AI play is an optional extra
  // that needs `python tetris/server.py` running on a GPU machine. Poll for
  // it so the button lights up the moment that server appears, with no
  // reload needed, and stays honest about the fact when it's not there.

  let aiBackendUp = false;
  async function pollAiBackend() {
    const up = await window.checkAiBackend();
    if (up !== aiBackendUp) {
      aiBackendUp = up;
      const btn = document.getElementById("ai-btn");
      btn.disabled = !up;
      btn.title = up ? "" : L.aiBackendMissing;
      if (!up && aiMode) setAiMode(false);
      if (!aiMode) {
        document.getElementById("ai-status").textContent = up ? "" : L.aiBackendMissing;
      }
    }
    setTimeout(pollAiBackend, 4000);
  }
  pollAiBackend();

  window.ROTATIONS_FOR = function (type) {
    return TetrisEngine ? enginePieceRotations(type) : null;
  };
  function enginePieceRotations(type) {
    // mirrors engine.js ROTATIONS for the mini preview canvases
    const base = {
      I: [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
      O: [[1, 1], [1, 1]],
      T: [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
      S: [[0, 1, 1], [1, 1, 0], [0, 0, 0]],
      Z: [[1, 1, 0], [0, 1, 1], [0, 0, 0]],
      J: [[1, 0, 0], [1, 1, 1], [0, 0, 0]],
      L: [[0, 0, 1], [1, 1, 1], [0, 0, 0]],
    };
    return base[type];
  }

  newGame();
})();
