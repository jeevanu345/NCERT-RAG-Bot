import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const python =
  process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
  });
  if (result.error || result.status !== 0) {
    console.error(
      result.error?.message || "Setup stopped. See the error above.",
    );
    process.exit(result.status || 1);
  }
}
run(process.env.PYTHON || "python3", ["-m", "venv", ".venv"]);
run(python, ["-m", "pip", "install", "-r", "requirements.txt"]);
run("npm", ["--prefix", "Full stack Project", "ci"]);
console.log(
  "\nSetup complete. Run npm run dev, then connect your NVIDIA key in Settings.",
);
