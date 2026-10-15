import { expect, test, type BrowserContext } from "@playwright/test";
import { encode } from "next-auth/jwt";
import path from "node:path";
import { providerAvailability, safeReturnTo } from "../../src/lib/auth-config";
import { createAppleClientSecret } from "../../scripts/apple-token.mjs";
import { exportPKCS8, generateKeyPair, jwtVerify } from "jose";

const secret = "ncert-e2e-only-secret-never-use-in-production-0001";
const accountId = "aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa";
// A real encrypted Auth.js session signed by the isolated test server's secret.
// No app authentication bypass or real provider credentials are used.
async function session(
  context: BrowserContext,
  origin: string,
  id = accountId,
  signingSecret = secret,
) {
  const value = await encode({
    secret: signingSecret,
    salt: "authjs.session-token",
    maxAge: 3600,
    token: {
      workspaceId: id,
      name: "Ada Student",
      email: "ada@example.test",
      sub: "test-subject",
    },
  });
  await context.addCookies([
    {
      name: "authjs.session-token",
      value,
      url: origin,
      httpOnly: true,
      sameSite: "Lax",
    },
  ]);
}

test("provider readiness and return destinations reject unsafe values", async () => {
  expect(providerAvailability({}).every((p) => !p.enabled)).toBe(true);
  const env = {
    AUTH_SECRET: secret,
    AUTH_GOOGLE_ID: "id",
    AUTH_GOOGLE_SECRET: "secret",
  };
  expect(providerAvailability(env).map((p) => p.enabled)).toEqual([
    true,
    false,
    false,
  ]);
  expect(
    providerAvailability({ ...env, AUTH_SECRET: "short" }).every(
      (p) => !p.enabled,
    ),
  ).toBe(true);
  for (const value of [
    "https://evil.test",
    "//evil.test",
    "/app\\evil.test",
    "/application",
    "/app\n/evil",
    null,
    ["/app"],
  ])
    expect(safeReturnTo(value)).toBe("/app");
  expect(safeReturnTo("/app?topic=current")).toBe("/app?topic=current");
});

test("Apple client secret is a verifiable ES256 JWT with bounded expiry", async () => {
  const { privateKey, publicKey } = await generateKeyPair("ES256", {
    extractable: true,
  });
  const pem = await exportPKCS8(privateKey);
  const now = Math.floor(Date.now() / 1000);
  const { secret: token } = await createAppleClientSecret({
    teamId: "TEAM123",
    clientId: "com.example.web",
    keyId: "KEY123",
    privateKey: pem,
    days: 180,
    now,
  });
  const { payload, protectedHeader } = await jwtVerify(token, publicKey, {
    issuer: "TEAM123",
    audience: "https://appleid.apple.com",
    subject: "com.example.web",
  });
  expect(protectedHeader).toMatchObject({ alg: "ES256", kid: "KEY123" });
  expect(payload.exp! - payload.iat!).toBe(180 * 86400);
  await expect(
    createAppleClientSecret({
      teamId: "TEAM",
      clientId: "com.example",
      keyId: "KEY",
      privateKey: pem,
      days: 181,
    }),
  ).rejects.toThrow("180 days");
});

test("sign-in offers configured providers and guest access without exposing credentials", async ({
  page,
}) => {
  await page.goto("/sign-in?returnTo=https://evil.test");
  await expect(
    page.getByRole("button", { name: "Continue with Google" }),
  ).toBeEnabled();
  await expect(
    page.getByRole("button", { name: "Continue with Microsoft", exact: false }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Continue with Apple", exact: false }),
  ).toBeDisabled();
  await expect(page.locator('input[name="returnTo"]').first()).toHaveValue(
    "/app",
  );
  const response = await page.request.get("/api/auth/providers");
  expect(response.status()).toBe(200);
  const providers = await response.json();
  expect(providers.google.callbackUrl).toMatch(
    /\/api\/auth\/callback\/google$/,
  );
  expect(JSON.stringify(providers)).not.toContain("test-google-secret");
  const csrf = await page.request.get("/api/auth/csrf");
  expect((await csrf.json()).csrfToken).toBeTruthy();
  const rejected = await page.request.post("/api/auth/signout", {
    form: { callbackUrl: "/app" },
    maxRedirects: 0,
  });
  expect(rejected.headers().location).toContain("MissingCSRF");
  await page.getByRole("link", { name: "Continue as a guest" }).click();
  await expect(page).toHaveURL(/\/app$/);
});

test("signed-in libraries and histories are separate from guests and other accounts", async ({
  page,
  context,
  browser,
  baseURL,
}) => {
  await session(context, baseURL!);
  await page.goto("/app");
  await page.getByRole("button", { name: "Open account menu" }).click();
  await expect(page.getByText("ada@example.test")).toBeVisible();
  await page.keyboard.press("Escape");
  await page
    .locator('input[aria-label="Attach PDFs"]')
    .setInputFiles(path.resolve("public/sample-study-notes.pdf"));
  await expect(page.getByRole("status")).toContainText("ready to study");
  const response = await page.request.get("/api/documents");
  const { documents } = await response.json();
  expect(documents).toHaveLength(1);
  const other = await browser.newContext({ baseURL });
  try {
    await other.addCookies([
      { name: "ncert-workspace", value: accountId, url: baseURL! },
    ]);
    const guest = await other.request.get("/api/documents", {
      headers: { "X-Workspace-ID": accountId, "X-Workspace-Kind": "account" },
    });
    expect((await guest.json()).documents).toEqual([]);
    await session(other, baseURL!, "bbbbbbbb-bbbb-4bbb-abbb-bbbbbbbbbbbb");
    expect(
      (await (await other.request.get("/api/documents")).json()).documents,
    ).toEqual([]);
    expect(
      (
        await other.request.delete(`/api/documents/${documents[0].id}`)
      ).status(),
    ).toBe(404);
  } finally {
    await other.close();
  }
  await page.evaluate((id) => {
    const messages = [{ role: "user", content: "Account-only question" }];
    localStorage.setItem(
      `ncert-sessions-v1:${id}`,
      JSON.stringify([{ id: "private", title: "Private session", messages }]),
    );
  }, accountId);
  await page.reload();
  await expect(page.locator(".user-message")).toContainText(
    "Account-only question",
  );
  await context.clearCookies();
  await page.reload();
  await expect(page.locator(".user-message")).not.toBeVisible();
});

test("sign-out clears cookies, account chat cache, and remembered key", async ({
  page,
  context,
  baseURL,
}) => {
  await session(context, baseURL!);
  await page.goto("/sign-out");
  await page.setViewportSize({ width: 320, height: 850 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Pause here. Pick up later.",
  );
  await page.evaluate((id) => {
    localStorage.setItem(`ncert-sessions-v1:${id}`, "private");
    sessionStorage.setItem(`ncert-key-v1:${id}`, "nvapi-test-placeholder");
  }, accountId);
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page).toHaveURL(/\/sign-in\?signedOut=1$/);
  await expect(page.getByRole("status")).toContainText("You’re signed out");
  expect(
    await page.evaluate(
      (id) => ({
        chats: localStorage.getItem(`ncert-sessions-v1:${id}`),
        key: sessionStorage.getItem(`ncert-key-v1:${id}`),
      }),
      accountId,
    ),
  ).toEqual({ chats: null, key: null });
  expect(
    (await context.cookies()).some((c) => c.name === "authjs.session-token"),
  ).toBe(false);
  await page.goto("/app");
  await expect(
    page.getByRole("link", { name: "Sign in", exact: true }),
  ).toBeVisible();
});

test("invalid encrypted sessions cannot authenticate and OAuth errors stay readable", async ({
  page,
  context,
  baseURL,
}) => {
  await session(
    context,
    baseURL!,
    accountId,
    "incorrect-secret-that-cannot-authenticate-1234",
  );
  await page.goto("/app");
  await expect(
    page.getByRole("link", { name: "Sign in", exact: true }),
  ).toBeVisible();
  await page.goto("/auth/error?error=AccessDenied");
  await expect(
    page.getByText("Sign-in was cancelled or access was declined.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Back to sign in" }).click();
  await expect(page).toHaveURL(/\/sign-in$/);
});

test("required sign-in fails closed when no providers are configured", async ({
  page,
}) => {
  const origin = "http://127.0.0.1:3200";
  await page.goto(`${origin}/app`);
  await expect(page).toHaveURL(/\/sign-in\?error=SessionRequired$/);
  await expect(
    page.getByRole("link", { name: "Continue as a guest" }),
  ).toHaveCount(0);
  for (const provider of ["Google", "Microsoft", "Apple"])
    await expect(
      page.getByRole("button", {
        name: `Continue with ${provider}`,
        exact: false,
      }),
    ).toBeDisabled();
  expect((await page.request.get(`${origin}/api/documents`)).status()).toBe(
    401,
  );
  expect(
    (
      await page.request.post(`${origin}/api/connection`, {
        data: { model: "meta/llama-3.3-70b-instruct" },
      })
    ).status(),
  ).toBe(401);
  expect(
    (await page.request.get(`${origin}/api/auth/providers`)).status(),
  ).toBe(503);
  expect((await page.request.get(`${origin}/api/health`)).status()).toBe(200);
});
