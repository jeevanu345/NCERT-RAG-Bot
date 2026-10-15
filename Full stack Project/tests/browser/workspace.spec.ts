import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { readFile } from "node:fs/promises";

const pdf = path.resolve("public/sample-study-notes.pdf");
const fakeKey = "nvapi-browser-test-placeholder";
const model = "meta/llama-3.3-70b-instruct";
const answer = {
  answer: "Electric current is charge flowing per unit time. [1]",
  model,
  sources: [
    {
      citation: 1,
      document_id: "test",
      name: "sample-study-notes.pdf",
      page: 1,
      excerpt:
        "Electric current is the rate of flow of electric charge. I = Q / t.",
      cited: true,
    },
  ],
};

async function open(page: Page) {
  await page.goto("/app");
  await expect(
    page.getByRole("button", { name: "Find your “aha” moment", exact: false }),
  ).toBeEnabled();
}
async function connect(page: Page) {
  // Only the provider response is simulated. PDF processing and storage use the real backend.
  await page.route("**/api/connection", (route) => {
    expect(route.request().headers().authorization).toBe(`Bearer ${fakeKey}`);
    return route.fulfill({ json: { connected: true, model } });
  });
  await page
    .getByRole("button", { name: "Connect NVIDIA", exact: true })
    .click();
  await page.getByLabel("API key", { exact: true }).fill(fakeKey);
  await page.getByRole("button", { name: "Test & save connection" }).click();
  await expect(
    page.getByText("Connected. You’re ready to study."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close dialog" }).click();
}
async function upload(page: Page) {
  await page.locator('input[aria-label="Attach PDFs"]').setInputFiles(pdf);
  await expect(page.getByRole("status")).toContainText("ready to study");
}
async function send(page: Page, question = "What is electric current?") {
  await page
    .getByRole("textbox", { name: "Ask your textbook a question" })
    .fill(question);
  await page.getByRole("button", { name: "Send question" }).click();
}

test("onboarding, starter actions, keyboard shortcut, and key validation", async ({
  page,
}) => {
  await open(page);
  await expect(
    page.getByRole("heading", { name: "Make sense of your next chapter." }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Keep the important bits", exact: false })
    .click();
  await expect(
    page.getByRole("button", { name: "Revise", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(
    page.getByRole("textbox", { name: "Ask your textbook a question" }),
  ).toHaveValue(/Summarize/);
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByLabel("API key", { exact: true }).fill("wrong-key");
  await page.getByRole("button", { name: "Test & save connection" }).click();
  await expect(page.getByRole("alert")).toContainText("nvapi-");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  await page.keyboard.press("Control+k");
  await expect(
    page.getByRole("textbox", { name: "Ask your textbook a question" }),
  ).toHaveValue("");
});

test("real PDF upload, deduplication, textbook search, scope, and removal", async ({
  page,
}) => {
  await open(page);
  await upload(page);
  await upload(page);
  await page.getByRole("button", { name: /Textbook library 1/ }).click();
  await expect(page.getByText("Ready to study", { exact: true })).toHaveCount(
    1,
  );
  await page
    .getByRole("textbox", { name: "Search textbooks" })
    .fill("unmatched");
  await expect(page.getByText("No textbooks match your search.")).toBeVisible();
  await page.getByRole("textbox", { name: "Search textbooks" }).fill("");
  await page.getByRole("button", { name: "Study this" }).click();
  await expect(page.getByLabel("Textbooks used for answers")).not.toHaveValue(
    "",
  );
  await page.getByRole("button", { name: /Textbook library 1/ }).click();
  await page
    .getByRole("button", { name: "Remove sample-study-notes.pdf", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Remove textbook", exact: true })
    .click();
  await expect(
    page.getByText("Your shelf is waiting for its first book."),
  ).toBeVisible();
});

test("invalid PDFs produce a real, recoverable error", async ({ page }) => {
  await open(page);
  await page.locator('input[aria-label="Attach PDFs"]').setInputFiles({
    name: "broken.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("invalid pdf"),
  });
  await expect(page.locator(".error-banner")).toContainText("not a valid PDF");
  await page.getByRole("button", { name: "Dismiss error" }).click();
  await upload(page);
});

test("cited answers, save, export, search history, and persistence", async ({
  page,
}) => {
  await open(page);
  await connect(page);
  await upload(page);
  await page.route("**/api/chat", (route) => {
    const body = route.request().postDataJSON();
    expect(body.model).toBe(model);
    expect(body.mode).toBe("explain");
    expect(body.messages[0].content).toBe("What is electric current?");
    return route.fulfill({ json: answer });
  });
  await send(page);
  await expect(page.getByText(answer.answer)).toBeVisible();
  await page.getByText("1 textbook passage", { exact: false }).click();
  await page.getByText("PDF page 1 · Cited in answer").click();
  await expect(
    page.getByText(answer.sources[0].excerpt, { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Save answer", exact: true }).click();
  await page
    .getByRole("button", { name: "Saved answers", exact: true })
    .click();
  await expect(page.getByText(answer.answer)).toBeVisible();
  await page.getByRole("button", { name: "Study space", exact: true }).click();
  const download = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export session as Markdown" })
    .click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(/\.md$/);
  expect(await readFile((await file.path())!, "utf8")).toContain("PDF page 1");
  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("ncert-sessions-v1")))
    .toContain("electric current");
  const storage = await page.evaluate(() =>
    JSON.stringify({ ...localStorage, ...sessionStorage }),
  );
  expect(storage).not.toContain(fakeKey);
  await page.reload();
  await expect(page.getByText(answer.answer)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Connect NVIDIA", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Search study sessions").fill("nothing matches");
  await expect(page.getByText("No matching sessions.")).toBeVisible();
});

test("flashcards reveal, navigate, remember, and restart", async ({ page }) => {
  await open(page);
  await connect(page);
  await upload(page);
  await page.route("**/api/chat", (route) =>
    route.fulfill({
      json: {
        ...answer,
        cards: [
          { question: "What is current?", answer: "Charge per unit time. [1]" },
          { question: "Unit of current?", answer: "The ampere. [1]" },
        ],
      },
    }),
  );
  await page.getByRole("button", { name: "Flashcards", exact: true }).click();
  await send(page, "Create flashcards about current.");
  await expect(
    page.getByText("What is current?", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Reveal answer" }).click();
  await expect(
    page.getByText("Charge per unit time. [1]", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "I know this" }).click();
  await expect(page.getByText("1 remembered")).toBeVisible();
  await page.getByRole("button", { name: "Next flashcard" }).click();
  await expect(
    page.getByText("Unit of current?", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Restart flashcard review" }).click();
  await expect(page.getByText("0 remembered")).toBeVisible();
});

test("provider failure can be retried without duplicating the question", async ({
  page,
}) => {
  await open(page);
  await connect(page);
  await upload(page);
  let attempts = 0;
  await page.route("**/api/chat", (route) =>
    ++attempts === 1
      ? route.fulfill({
          status: 429,
          json: {
            detail: "NVIDIA request limit reached. Please wait and retry.",
          },
        })
      : route.fulfill({ json: answer }),
  );
  await send(page);
  await expect(page.locator(".error-banner")).toContainText("request limit");
  await page.getByRole("button", { name: "Retry question" }).click();
  await expect(page.getByText(answer.answer)).toBeVisible();
  await expect(page.locator(".user-message")).toHaveCount(1);
});

test("stop restores retry and session deletion removes saved content", async ({
  page,
}) => {
  await open(page);
  await connect(page);
  await upload(page);
  await page.route("**/api/chat", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    await route.fulfill({ json: answer }).catch(() => {});
  });
  await send(page);
  await page.getByRole("button", { name: "Stop answer" }).click();
  await expect(
    page.getByRole("button", { name: "Retry question" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Delete session: What is electric current?" })
    .click();
  await page
    .getByRole("button", { name: "Delete session", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Make sense of your next chapter." }),
  ).toBeVisible();
});

test("mobile navigation, sample notes, dialog keyboard, and responsive layout", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Open navigation" }).click();
  await page.getByRole("button", { name: /Textbook library 0/ }).click();
  await page.getByRole("button", { name: "Try sample science notes" }).click();
  await expect(
    page.getByRole("heading", {
      name: "Sample science notes.pdf",
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Study this" }).click();
  await page
    .getByRole("button", { name: "Connect NVIDIA", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  expect(
    await page
      .getByRole("dialog")
      .evaluate((el) => el.getBoundingClientRect().width < innerWidth),
  ).toBe(true);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).not.toBeVisible();
  for (const width of [320, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 850 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});

test("study, library, and settings pass automated accessibility checks", async ({
  page,
}) => {
  const { default: AxeBuilder } = await import("@axe-core/playwright");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await open(page);
  const violations = [];
  for (const mode of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme: mode });
    await open(page);
    // Check the settled, enabled state rather than the initial loading fade.
    await expect(
      page.getByRole("button", { name: /New study session/ }),
    ).toHaveCSS("opacity", "1");
    for (const screen of ["study", "library", "settings"]) {
      if (screen === "library")
        await page.getByRole("button", { name: /Textbook library 0/ }).click();
      if (screen === "settings") {
        await page
          .getByRole("button", { name: "Connect NVIDIA", exact: true })
          .click();
        await page
          .getByLabel("API key", { exact: true })
          .fill("nvapi-accessibility-placeholder");
        await page.getByRole("button", { name: "Show API key" }).click();
      }
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      violations.push(
        ...results.violations.map((v) => ({
          screen,
          mode,
          id: v.id,
          nodes: v.nodes.map((n) => ({
            target: n.target,
            summary: n.failureSummary,
          })),
        })),
      );
    }
  }
  expect(violations).toEqual([]);
});

test("corrupt stored flashcards cannot break the workspace", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem(
      "ncert-sessions-v1",
      JSON.stringify([
        {
          id: "bad-session",
          title: "Corrupt session",
          messages: [
            {
              role: "assistant",
              content: "bad data",
              cards: [{ question: { bad: true }, answer: null }],
            },
          ],
        },
      ]),
    ),
  );
  await open(page);
  await expect(
    page.getByRole("heading", { name: "Make sense of your next chapter." }),
  ).toBeVisible();
});

test("proxy rejects cross-origin writes and isolates browser libraries", async ({
  page,
  browser,
}) => {
  await open(page);
  await upload(page);
  const forbidden = await page.request.post(
    "/api/upload?filename=foreign.pdf",
    {
      headers: {
        Origin: "https://unrelated.invalid",
        "Content-Type": "application/pdf",
      },
      data: await readFile(pdf),
    },
  );
  expect(forbidden.status()).toBe(403);
  const other = await browser.newContext({
    baseURL: new URL(page.url()).origin,
  });
  try {
    const library = await other.request.get("/api/documents");
    expect(library.status()).toBe(200);
    expect((await library.json()).documents).toEqual([]);
  } finally {
    await other.close();
  }
  const missingName = await page.request.post("/api/upload", { data: "empty" });
  expect(missingName.status()).toBe(400);
});
