import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/browser",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
      },
    },
  ],
  webServer: [
    {
      command: "node ../scripts/dev.mjs --production",
      url: process.env.E2E_BASE_URL || "http://127.0.0.1:3100",
      // Isolated credentials, ports, and data. Never reuse a developer's signed-in app.
      reuseExistingServer: false,
      env: {
        PORT: "3100",
        API_PORT: "8100",
        NCERT_DATA_DIR: "../.data/e2e",
        AUTH_URL: "http://127.0.0.1:3100",
        AUTH_SECRET: "ncert-e2e-only-secret-never-use-in-production-0001",
        AUTH_GOOGLE_ID: "test-google-client",
        AUTH_GOOGLE_SECRET: "test-google-secret",
        AUTH_MICROSOFT_ENTRA_ID_ID: "",
        AUTH_MICROSOFT_ENTRA_ID_SECRET: "",
        AUTH_APPLE_ID: "",
        AUTH_APPLE_SECRET: "",
        AUTH_REQUIRED: "false",
        NVIDIA_API_KEY: "",
      },
      timeout: 30000,
    },
    {
      command: "node ../scripts/dev.mjs --production",
      url: "http://127.0.0.1:3200",
      reuseExistingServer: false,
      timeout: 30000,
      env: {
        PORT: "3200",
        API_PORT: "8200",
        NCERT_DATA_DIR: "../.data/e2e-required",
        AUTH_URL: "http://127.0.0.1:3200",
        AUTH_SECRET: "",
        AUTH_REQUIRED: "true",
        AUTH_GOOGLE_ID: "",
        AUTH_GOOGLE_SECRET: "",
        AUTH_MICROSOFT_ENTRA_ID_ID: "",
        AUTH_MICROSOFT_ENTRA_ID_SECRET: "",
        AUTH_APPLE_ID: "",
        AUTH_APPLE_SECRET: "",
        NVIDIA_API_KEY: "",
      },
    },
  ],
});
