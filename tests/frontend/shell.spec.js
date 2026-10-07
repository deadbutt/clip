const { test, expect } = require('@playwright/test');
const { installApiFixture, openEditor } = require('./fixtures');

test('home, import shortcuts and task filters share the existing projects', async ({ page }, testInfo) => {
  await installApiFixture(page);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#hubTotal')).toHaveText('1');
  await expect(page.locator('#hubEditable')).toHaveText('1');
  await expect(page.locator('.upload-card')).toBeHidden();
  await page.screenshot({ path: testInfo.outputPath('home.png'), fullPage: true });
  await page.locator('.hub-entry[data-shell-create="url"]').click();
  await expect(page.locator('#urlTab')).toBeVisible();
  await expect(page.locator('.jobs-card')).toBeHidden();
  await page.screenshot({ path: testInfo.outputPath('create.png'), fullPage: true });
  await page.locator('.shell-nav[data-shell-page="tasks"]').click();
  await page.locator('#hubTaskSearch').fill('missing');
  await expect(page.locator('.task-item')).toHaveCount(0);
  await expect(page.locator('.hub-empty')).toContainText('没有匹配');
  await page.locator('#hubTaskSearch').fill('fixture');
  await expect(page.locator('.task-item')).toHaveCount(1);
  await page.locator('#hubTaskFilter').selectOption('running');
  await expect(page.locator('.task-item')).toHaveCount(0);
  await page.locator('#hubTaskFilter').selectOption('editable');
  await expect(page.locator('.task-item')).toHaveCount(1);
  await page.screenshot({ path: testInfo.outputPath('tasks.png'), fullPage: true });
  await page.locator('.task-item').click();
  await expect(page.locator('#workbench')).toBeVisible();
  await expect(page.locator('.app-sidebar')).toBeHidden();
  await page.locator('#backToTasks').click();
  await expect(page.locator('.app-sidebar')).toBeVisible();
  expect(errors).toEqual([]);
});

test('small screens keep navigation and import reachable without horizontal overflow', async ({ page }, testInfo) => {
  await installApiFixture(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.hub-entry').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: testInfo.outputPath('mobile-home.png'), fullPage: true });
  await page.locator('.hub-entry[data-shell-create="upload"]').click();
  await expect(page.locator('#uploadTab')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
