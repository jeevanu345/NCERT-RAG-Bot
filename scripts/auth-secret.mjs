import { randomBytes } from "node:crypto";
import { readFile, writeFile, chmod } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const file = fileURLToPath(
  new URL("../Full stack Project/.env.local", import.meta.url),
);
let contents = "";
try {
  contents = await readFile(file, "utf8");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
if (
  /^[ \t]*(?:export[ \t]+)?AUTH_SECRET[ \t]*=[ \t]*["']?[^\s"'#]+/m.test(
    contents,
  )
) {
  console.log("AUTH_SECRET is already set. It was left unchanged.");
} else {
  const entry = `AUTH_SECRET=${randomBytes(48).toString("base64url")}`;
  contents = /^[ \t]*AUTH_SECRET[ \t]*=.*$/m.test(contents)
    ? contents.replace(/^[ \t]*AUTH_SECRET[ \t]*=.*$/m, entry)
    : `${contents.trimEnd()}\n\n# Encrypted OAuth sessions. Keep this private and stable.\n${entry}\n`;
  await writeFile(file, contents, { mode: 0o600 });
  await chmod(file, 0o600);
  console.log(
    "Generated AUTH_SECRET in Full stack Project/.env.local. No secret was printed.",
  );
}
