const { defineConfig } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');

const installedChrome = process.platform === 'win32' && [
  process.env.ProgramFiles,
  process.env['ProgramFiles(x86)'],
  process.env.LOCALAPPDATA,
].filter(Boolean).some((root) => fs.existsSync(path.join(root, 'Google', 'Chrome', 'Application', 'chrome.exe')));

module.exports = defineConfig({
  testDir: './tests/frontend',
  timeout: 20_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [['line']] : [['list']],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    channel: process.env.PLAYWRIGHT_CHANNEL || (installedChrome ? 'chrome' : undefined),
    viewport: { width: 1400, height: 900 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'python tests/frontend/fixture_server.py --port 4173',
    url: 'http://127.0.0.1:4173/',
    reuseExistingServer: !process.env.CI,
    timeout: 10_000,
  },
});
