# PACT Tetris — desktop app

A packaged, double-click version of [`../tetris`](../tetris): manual play works
completely standalone, no Python needed. AI play is optional and looks for a
local backend at `http://127.0.0.1:8848` — if it's not running, the "Let PACT
play" button just stays greyed out with an explanation, rather than erroring.

## Run a build someone already made you

**Windows** — run the installer (`PACT Tetris Setup *.exe`) or just unzip and
run the portable `.exe`. No other setup needed.

**macOS** — unzip `PACT Tetris-*-mac.zip`, drag `PACT Tetris.app` to
Applications. **First launch only:** macOS will refuse to open it
("`PACT Tetris` is damaged and can't be opened" or similar) because this
build isn't signed with an Apple Developer certificate. To open it anyway:
right-click (or Control-click) the app → **Open** → **Open** in the dialog
that appears. You only need to do this once; after that it opens normally.
(If that doesn't work: System Settings → Privacy & Security → scroll to the
blocked-app notice → **Open Anyway**.)

## Enable AI play

On a machine with an NVIDIA GPU, from the repo root:

```bash
python tetris/server.py
```

Leave that running, then open (or switch to) the desktop app — the "Let PACT
play" button enables itself automatically within a few seconds, no restart
needed. It stays a separate process on purpose: the model backend is a
multi-GB PyTorch/CUDA environment that has no reason to ship inside a
Windows/Mac game installer that most people will only ever use for manual
play.

## Building it yourself

```bash
cd desktop
npm install
npm run start      # dev mode, opens a window straight away

npm run dist:win    # -> dist/PACT Tetris Setup *.exe, dist/PACT Tetris *.exe (portable)
npm run dist:mac    # -> dist/PACT Tetris-*-mac.zip (unsigned; built from Windows, see above)
npm run dist:all    # both
```

`app/` is the single source of truth for the game itself (`server.py` also
serves straight from it) — edit there, not in a copy.

### Building the Mac version

electron-builder flatly refuses to build a macOS target from a non-Mac host
("Build for macOS is supported only on macOS") — this isn't a signing
limitation, it's checked before that. So `npm run dist:mac` only works when
run **on an actual Mac** (or a macOS CI runner, e.g. GitHub Actions'
`macos-latest`). The Windows build has no such restriction and was built and
smoke-tested from this Windows machine.

On a Mac, from the repo root:

```bash
cd desktop
npm install
npm run dist:mac    # -> dist/PACT Tetris-*-mac.zip
```

That `.app` will still be unsigned (see the Gatekeeper note above) unless you
also have an Apple Developer account — add `CSC_LINK` / `CSC_KEY_PASSWORD`
(or `APPLE_ID` / `APPLE_APP_SPECIFIC_PASSWORD` for notarization) as
environment variables and electron-builder signs automatically; see the
[electron-builder code-signing docs](https://www.electron.build/code-signing).
