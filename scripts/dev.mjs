import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const app = path.join(root, "Full stack Project");
const python =
  process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
if (!existsSync(python) || !existsSync(path.join(app, "node_modules"))) {
  console.error("Install dependencies first: npm run setup");
  process.exit(1);
}
const port = process.env.PORT || "3000";
const apiPort = process.env.API_PORT || "8000";
const production = process.argv.includes("--production");
const children = [];
let closing = false;
function stop(code = 0) {
  if (closing) return;
  closing = true;
  for (const child of children) if (!child.killed) child.kill("SIGTERM");
  const timeout = setTimeout(() => {
    for (const child of children)
      if (child.exitCode === null) child.kill("SIGKILL");
    process.exit(code);
  }, 2500);
  timeout.unref();
  Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          if (child.exitCode !== null) resolve();
          else child.once("exit", resolve);
        }),
    ),
  ).then(() => process.exit(code));
}
function start(command, args, extraEnv = {}) {
  const child = spawn(command, args, {
    cwd: app,
    stdio: "inherit",
    env: { ...process.env, ...extraEnv },
    shell: process.platform === "win32",
  });
  children.push(child);
  child.on("error", (error) => {
    console.error(error.message);
    stop(1);
  });
  child.on("exit", (code) => {
    if (!closing) stop(code || 0);
  });
}
start(python, [
  "-m",
  "uvicorn",
  "api.index:app",
  "--host",
  "127.0.0.1",
  "--port",
  apiPort,
]);
start(
  process.execPath,
  [
    path.join(app, "node_modules/next/dist/bin/next"),
    production ? "start" : "dev",
    "--hostname",
    "127.0.0.1",
    "--port",
    port,
  ],
  { PYTHON_API_BASE_URL: `http://127.0.0.1:${apiPort}` },
);
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
console.log(
  `\nNCERT Tutor: http://127.0.0.1:${port}\nBoth services are local. Press Ctrl+C to stop.\n`,
);
