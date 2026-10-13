const { test, expect } = require('@playwright/test');
const { installApiFixture, openEditor } = require('./fixtures');

test('Bilibili draft saves metadata and submit progress links to the returned稿件', async ({ page }) => {
  await installApiFixture(page);
  const draft = {
    title: 'fixture.mp4', description: '', tags: [], tid: 0, copyright: 'original',
    source: '', dynamic: '', upload_variant: 'original', status: 'draft',
    available_variants: ['original'], cover_available: true,
  };
  let savedPayload;
  let pollCount = 0;
  await page.route('**/api/bilibili/auth', (route) => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ logged_in: true, account: 'test-account' }),
  }));
  await page.route('**/api/bilibili/categories', (route) => route.fulfill({
    status: 200, contentType: 'application/json', body: JSON.stringify({ categories: [{ tid: 12, name: '知识 · 科普' }] }),
  }));
  await page.route('**/api/jobs/job-1/thumbnail', (route) => route.fulfill({
    status: 200, contentType: 'image/png', body: Buffer.from('fake-png'),
  }));
  await page.route('**/api/jobs/job-1/publish', async (route) => {
    if (route.request().method() === 'PUT') {
      savedPayload = route.request().postDataJSON();
      Object.assign(draft, savedPayload, { updated_at: 1 });
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(draft) });
    }
    pollCount += 1;
    const response = pollCount > 1 ? {
      ...draft, status: 'submitted', phase: 'complete', progress: 1,
      bvid: 'BV1test000001', url: 'https://www.bilibili.com/video/BV1test000001',
    } : draft;
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(response) });
  });
  await page.route('**/api/jobs/job-1/publish/submit', async (route) => {
    savedPayload = route.request().postDataJSON();
    Object.assign(draft, savedPayload);
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ...draft, status: 'uploading', phase: 'uploading', progress: 0.2 }) });
  });

  await openEditor(page);
  await page.locator('#openPublish').click();
  await expect(page.locator('#bilibiliAuthStatus')).toContainText('test-account');
  await expect(page.locator('#publishCategory')).toContainText('知识 · 科普');
  await page.locator('#publishTitle').fill('我的测试投稿');
  await page.locator('#publishDescription').fill('简介内容');
  await page.locator('#publishTags').fill('字幕, 转录');
  await page.locator('#publishCategory').selectOption('12');
  await page.locator('#publishCopyright').selectOption('repost');
  await expect(page.locator('#publishSource')).toBeVisible();
  await page.locator('#publishSource').fill('https://source.example/video');
  await page.locator('#savePublish').click();
  await expect(page.locator('#publishStatus')).toContainText('本地草稿已保存');
  expect(savedPayload).toMatchObject({
    title: '我的测试投稿', tags: ['字幕', '转录'], tid: 12, copyright: 'repost',
    source: 'https://source.example/video', upload_variant: 'original',
  });

  await page.locator('#submitPublish').click();
  await expect(page.locator('#publishStatus')).toContainText('20%');
  await expect(page.locator('#publishStatus')).toContainText('已提交到 B 站');
  await expect(page.locator('#publishResultLink')).toHaveAttribute('href', 'https://www.bilibili.com/video/BV1test000001');
});
