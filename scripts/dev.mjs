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
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());
start(python, [
  "-m",
  "uvicorn",
  "api.index:app",
  "--host",
  "127.0.0.1",
  "--port",
  apiPort,
]);
// Open the UI only once its API is ready, including on a cold CI runner.
const deadline = Date.now() + 15000;
let ready = false;
while (!closing && Date.now() < deadline) {
  try {
    const response = await fetch(`http://127.0.0.1:${apiPort}/api/health`, {
      signal: AbortSignal.timeout(500),
    });
    const health = await response.json();
    if (
      response.ok &&
      health.status === "ok" &&
      health.provider === "NVIDIA NIM"
    ) {
      ready = true;
      break;
    }
  } catch {
    // The Python process is still starting; a failed child exits through stop().
  }
  await new Promise((resolve) => setTimeout(resolve, 100));
}
if (!closing && ready) {
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
  console.log(
    `\nNCERT Tutor: http://127.0.0.1:${port}\nBoth services are local. Press Ctrl+C to stop.\n`,
  );
} else if (!closing) {
  console.error(
    "The Python API did not become ready. Check its startup error above.",
  );
  stop(1);
}
