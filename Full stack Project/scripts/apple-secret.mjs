import { readFile, mkdir, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { pathToFileURL } from "node:url";
import path from "node:path";
import { createAppleClientSecret } from "./apple-token.mjs";

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  try {
    const { values } = parseArgs({
      options: {
        "team-id": { type: "string" },
        "client-id": { type: "string" },
        "key-id": { type: "string" },
        "private-key": { type: "string" },
        days: { type: "string", default: "180" },
        out: { type: "string", default: ".secrets/apple-client-secret.env" },
      },
    });
    if (!values["private-key"])
      throw new Error(
        "Pass --private-key /private/path/AuthKey.p8. See docs/OAUTH_SETUP.md.",
      );
    const privateKey = await readFile(values["private-key"], "utf8");
    const { secret, expires } = await createAppleClientSecret({
      teamId: values["team-id"],
      clientId: values["client-id"],
      keyId: values["key-id"],
      privateKey,
      days: Number(values.days),
    });
    const output = path.resolve(values.out);
    await mkdir(path.dirname(output), { recursive: true, mode: 0o700 });
    await writeFile(
      output,
      `# Expires ${new Date(expires * 1000).toISOString()}\nAUTH_APPLE_SECRET=${secret}\n`,
      { flag: "wx", mode: 0o600 },
    );
    console.log(
      `Apple client secret saved privately to ${output}. Expires ${new Date(expires * 1000).toISOString()}. Copy its AUTH_APPLE_SECRET entry into .env.local; renew before expiry.`,
    );
  } catch (error) {
    // Do not log key material or signed tokens in error output.
    console.error(
      error.code === "EEXIST"
        ? "Output already exists. Choose a different --out path for the renewed secret."
        : "Could not generate the Apple secret. Check the IDs, readable PKCS#8 .p8 key, expiry (1–180 days), and writable output path. See docs/OAUTH_SETUP.md.",
    );
    process.exitCode = 1;
  }
}
