import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function appearance(page: Page) {
  await page.goto("/app");
  await page
    .getByRole("button", { name: "Connect NVIDIA", exact: true })
    .click();
  await page.getByRole("tab", { name: "Appearance" }).click();
}

test("landing previews, source disclosure, FAQ, and mobile navigation work", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "A clearer way to meet your next chapter.",
  );
  const preview = page.getByRole("group", {
    name: "Preview study modes",
    exact: true,
  });
  await preview.getByRole("button", { name: "Practice", exact: true }).click();
  await page
    .locator(".hero-product")
    .getByRole("button", { name: "2 A", exact: true })
    .click();
  await expect(page.locator(".hero-product .demo-feedback")).toContainText(
    "That’s it.",
  );
  await preview
    .getByRole("button", { name: "Flashcards", exact: true })
    .click();
  await page
    .locator(".hero-product")
    .getByRole("button", { name: "Reveal answer", exact: true })
    .click();
  await expect(
    page
      .locator(".hero-product")
      .getByRole("heading", { name: "V = I × R", exact: true }),
  ).toBeVisible();
  await page.locator(".hero-product summary").click();
  await expect(page.locator(".hero-product .demo-source > p")).toBeVisible();
  await page
    .getByText("What does signing in give me?", { exact: true })
    .click();
  await expect(
    page.getByText("Signing in links the textbook library", { exact: false }),
  ).toBeVisible();
  for (const width of [320, 390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Open menu", exact: true }).click();
  await page
    .getByRole("navigation", { name: "Main navigation" })
    .getByRole("link", { name: "How it works", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Open menu", exact: true }),
  ).toHaveAttribute("aria-expanded", "false");
  await page.getByRole("link", { name: "Find your starting point" }).click();
  await expect(page).toHaveURL(/\/app$/);
});

test("mode, palette, and fonts apply across routes, persist, and reset", async ({
  page,
}) => {
  await appearance(page);
  const html = page.locator("html");
  await page.getByRole("button", { name: "Dark", exact: true }).click();
  await expect(html).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "Ocean", exact: true }).click();
  await page.getByLabel("Body font", { exact: true }).selectOption("inter");
  await page.getByLabel("Heading font", { exact: true }).selectOption("modern");
  await expect(html).toHaveAttribute("data-palette", "ocean");
  await expect(page.locator("body")).toHaveCSS("font-family", /Inter Variable/);
  await expect(page.locator(".appearance-preview h3")).toHaveCSS(
    "font-family",
    /Inter Variable/,
  );
  await page.goto("/");
  await expect(html).toHaveAttribute("data-theme", "dark");
  await expect(html).toHaveAttribute("data-palette", "ocean");
  await expect(page.getByRole("heading", { level: 1 })).toHaveCSS(
    "font-family",
    /Inter Variable/,
  );
  await page.goto("/sign-in");
  await expect(page.locator("body")).toHaveCSS("font-family", /Inter Variable/);
  await page.reload();
  await expect(html).toHaveAttribute("data-font", "inter");
  await appearance(page);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(html).toHaveAttribute("data-palette", "forest");
  await expect(html).toHaveAttribute("data-font", "manrope");
  await expect(html).toHaveAttribute("data-heading", "editorial");
  await expect(
    page.getByRole("button", { name: "System", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(html).toHaveAttribute("data-theme", "dark");
  await page.emulateMedia({ colorScheme: "light" });
  await expect(html).toHaveAttribute("data-theme", "light");
});

test("all palettes and both modes pass appearance accessibility checks", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await appearance(page);
  const violations = [];
  for (const mode of ["Light", "Dark"]) {
    await page.getByRole("button", { name: mode, exact: true }).click();
    for (const palette of [
      "Forest",
      "Ocean",
      "Iris",
      "Rose",
      "Amber",
      "Slate",
    ]) {
      await page.getByRole("button", { name: palette, exact: true }).click();
      const result = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      violations.push(
        ...result.violations.map((v) => ({
          mode,
          palette,
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

test("public pages are accessible in both color modes and narrow layouts", async ({
  page,
}) => {
  const runtimeErrors: string[] = [];
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") runtimeErrors.push(message.text());
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  const violations = [];
  for (const mode of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme: mode as "light" | "dark" });
    for (const route of [
      "/",
      "/sign-in",
      "/sign-out",
      "/auth/error?error=AccessDenied",
      "/privacy",
    ]) {
      await page.goto(route);
      await expect(page.locator("html")).toHaveAttribute("data-theme", mode);
      const result = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
        .analyze();
      violations.push(
        ...result.violations.map((v) => ({
          route,
          mode,
          id: v.id,
          nodes: v.nodes.map((n) => ({
            target: n.target,
            summary: n.failureSummary,
          })),
        })),
      );
      await page.setViewportSize({ width: 320, height: 850 });
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await page.setViewportSize({ width: 1440, height: 900 });
    }
  }
  expect(violations).toEqual([]);
  expect(runtimeErrors).toEqual([]);
});

test("corrupt and unrecognized appearance settings recover safely", async ({
  page,
}) => {
  await page.addInitScript(() =>
    localStorage.setItem(
      "ncert-appearance-v1",
      '{"palette":"url(evil)","font":{},"heading":null}',
    ),
  );
  await appearance(page);
  await expect(page.locator("html")).toHaveAttribute("data-palette", "forest");
  await expect(page.locator("html")).toHaveAttribute("data-font", "manrope");
  await expect(page.locator("html")).toHaveAttribute(
    "data-heading",
    "editorial",
  );
});
