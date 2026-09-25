const { app, BrowserWindow, Menu, shell } = require("electron");
const path = require("path");

const START_PAGE = "index_en.html";

function createWindow() {
  const win = new BrowserWindow({
    width: 1040,
    height: 760,
    minWidth: 760,
    minHeight: 600,
    backgroundColor: "#0a0c11",
    title: "PACT plays Tetris",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.loadFile(path.join(__dirname, "app", START_PAGE));

  // Local-file navigation (the language switcher link) stays inside the
  // window; anything else opens in the user's real browser instead of
  // turning this game window into a general-purpose one.
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (!url.startsWith("file://")) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  const menu = Menu.buildFromTemplate([
    {
      label: "Game",
      submenu: [
        { label: "English", click: () => win.loadFile(path.join(__dirname, "app", "index_en.html")) },
        { label: "中文", click: () => win.loadFile(path.join(__dirname, "app", "index_zh.html")) },
        { type: "separator" },
        { role: "reload" },
        { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [{ role: "toggleDevTools" }, { role: "resetZoom" }, { role: "zoomIn" }, { role: "zoomOut" }],
    },
  ]);
  Menu.setApplicationMenu(menu);
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
