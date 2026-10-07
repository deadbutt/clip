const { test, expect } = require('@playwright/test');
const { installApiFixture } = require('./fixtures');

async function fixture(page, multiple = false) {
  const jobs = await installApiFixture(page);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const sub = { id: 'sub-1', platform: 'youtube', channel_id: 'UCfixture', channel_name: '测试频道',
    name: '', url: 'https://www.youtube.com/channel/UCfixture', enabled: true, initialized: true,
    interval_minutes: 10, mode: 'notify', transcribe: false, cookies_browser: 'firefox', last_checked: 100 };
  const state = { subscriptions: [sub], videos: [
    { id: 'youtube:new', platform: 'youtube', subscription_id: sub.id, title: '新投稿 <script>',
      url: 'https://www.youtube.com/watch?v=abcdefghijk', is_new: true, unread: true, state: 'pending' },
    { id: 'youtube:old', platform: 'youtube', subscription_id: sub.id, title: '历史投稿',
      url: 'https://www.youtube.com/watch?v=abcdefghijl', is_new: false, unread: false, state: 'history' },
  ], unread_count: 1, running: true, downloads: 0, creates: [] };
  if (multiple) {
    state.subscriptions.push({ ...sub, id: 'sub-2', platform: 'bilibili', name: '另一个 UP 主' });
    state.videos.push({ id: 'bilibili:new', platform: 'bilibili', subscription_id: 'sub-2', title: '另一个频道的新视频',
      thumbnail: 'https://example.test/cover.svg', url: 'https://www.bilibili.com/video/BVfixture', is_new: true, unread: true, state: 'pending' });
    state.unread_count = 2;
    await page.route('https://example.test/cover.svg', (route) => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#dcece8"/><circle cx="160" cy="90" r="35" fill="#89b7a7"/></svg>' }));
  }
  await page.route(/\/api\/(subscriptions|subscription-videos)(\/|$)/, async (route) => {
    const request = route.request();
    const path = decodeURIComponent(new URL(request.url()).pathname);
    const method = request.method();
    let result = state;
    if (path === '/api/subscriptions' && method === 'POST') {
      state.creates.push(request.postDataJSON());
      result = sub;
    } else if (method === 'PUT') {
      Object.assign(sub, request.postDataJSON());
      result = sub;
    } else if (path.endsWith('/download')) {
      state.downloads++;
      Object.assign(state.videos[0], { state: 'enqueued', job_id: 'job-1', job_status: 'downloaded', unread: false });
      state.unread_count = 0;
      Object.assign(jobs.job, { status: 'downloaded', download_only: true, progress: 1,
        input_path: 'D:/isolated/input.mkv', media_name: '新投稿' });
      result = jobs.job;
    } else if (path.endsWith('/read')) {
      state.videos.forEach((v) => { v.unread = false; });
      state.unread_count = 0;
    } else if (path.endsWith('/dismiss')) {
      Object.assign(state.videos[0], { state: 'ignored', unread: false });
      state.unread_count = 0;
    } else if (path.endsWith('/check')) result = { new_count: 0 };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(result) });
  });
  await page.goto('/');
  await page.locator('#openSubscriptions').click();
  await expect(page.locator('#subscriptionsView')).toBeVisible();
  await page.locator('#subscriptionVideoFilter').selectOption('new');
  await expect(page.locator('.subscription-card')).toHaveCount(multiple ? 2 : 1);
  return { state, errors };
}

test('updates filter, escaped titles, edit and pause work without leaving subscriptions', async ({ page }) => {
  const { errors } = await fixture(page);
  await expect(page.locator('.subscription-video')).toHaveCount(1);
  await expect(page.locator('.subscription-video-title')).toHaveText('新投稿 <script>');
  await page.locator('#subscriptionVideoFilter').selectOption('all');
  await expect(page.locator('.subscription-video')).toHaveCount(2);
  await page.locator('.subscription-card[data-subscription-id="sub-1"]').click();
  await page.locator('.channel-management summary').click();
  await page.getByRole('button', { name: '编辑', exact: true }).click();
  await page.locator('#subscriptionName').fill('我的频道');
  await page.locator('#subscriptionSubmit').click();
  await expect(page.locator('.subscription-card strong')).toHaveText('我的频道');
  await page.getByRole('button', { name: '暂停', exact: true }).click();
  await expect(page.locator('.subscription-card')).toContainText('已暂停');
  await page.evaluate(() => window.refreshJobs({ keepSelection: true, background: true }));
  await expect(page.locator('#subscriptionsView')).toBeVisible();
  await page.locator('.shell-nav[data-shell-page="tasks"]').click();
  await page.locator('.task-item[data-job-id="job-1"]').click();
  await expect(page.locator('#workbench')).toBeVisible();
  expect(errors).toEqual([]);
});

test('confirmation queues one download and downloaded video opens transcription draft', async ({ page }) => {
  const { state, errors } = await fixture(page);
  await page.getByRole('button', { name: '确认下载', exact: true }).click();
  await expect(page.getByRole('button', { name: '查看任务', exact: true })).toBeVisible();
  expect(state.downloads).toBe(1);
  await expect(page.locator('#subscriptionsView')).toBeVisible();
  await page.getByRole('button', { name: '查看任务', exact: true }).click();
  await expect(page.locator('#downloadedView')).toBeVisible();
  await expect(page.locator('#downloadedPath')).toHaveValue('D:/isolated/input.mkv');
  await expect(page.locator('#downloadedMediaLink')).toHaveAttribute('href', /kind=media/);
  await page.locator('#transcribeDownloaded').click();
  await expect(page.locator('#importView')).toBeVisible();
  await expect(page.locator('#rerunSource')).toContainText('新投稿');
  expect(errors).toEqual([]);
});

test('adding Bilibili defaults to notification and download only; ignored updates clear badge', async ({ page }) => {
  const { state, errors } = await fixture(page);
  await page.locator('#addSubscription').click();
  await page.locator('#subscriptionPlatform').selectOption('bilibili');
  await page.locator('#subscriptionSource').fill('2');
  await page.locator('#subscriptionSubmit').click();
  await expect(page.locator('#subscriptionsMessage')).toContainText('关注已保存');
  await expect(page.locator('#subscriptionDialog')).not.toBeVisible();
  expect(state.creates[0]).toMatchObject({ platform: 'bilibili', source: '2', mode: 'notify',
    interval_minutes: 10, transcribe: false, cookies_browser: 'firefox' });
  await page.getByRole('button', { name: '忽略', exact: true }).click();
  await expect(page.locator('.subscription-video')).toHaveCount(0);
  await expect(page.locator('#subscriptionBadge')).toBeHidden();
  expect(errors).toEqual([]);
});

test('inbox aggregates new posts, channel selection scopes videos and unread reminders persist until handled', async ({ page }, testInfo) => {
  const { errors } = await fixture(page, true);
  await expect(page.locator('#subscriptionDialog')).not.toBeVisible();
  await expect(page.locator('.subscription-video')).toHaveCount(2);
  await expect(page.locator('#subscriptionBadge')).toHaveText('2');
  await expect(page.locator('.subscription-card .subscription-unread-count')).toHaveCount(2);
  await page.screenshot({ path: testInfo.outputPath('subscriptions-inbox.png') });
  await page.locator('[data-subscription-id="sub-2"].subscription-card').click();
  await expect(page.locator('#subscriptionFeedTitle')).toHaveText('另一个 UP 主');
  await expect(page.locator('.subscription-video')).toHaveCount(1);
  await expect(page.locator('.subscription-video-title')).toHaveText('另一个频道的新视频');
  await expect(page.locator('img.subscription-thumbnail')).toBeVisible();
  await expect(page.locator('#subscriptionBadge')).toHaveText('2');
  await page.screenshot({ path: testInfo.outputPath('subscriptions-channel.png') });
  await page.locator('#subscriptionSearch').fill('测试');
  await expect(page.locator('.subscription-card')).toHaveCount(1);
  await page.locator('#allSubscriptionUpdates').click();
  await expect(page.locator('.subscription-video')).toHaveCount(2);
  await page.locator('#readSubscriptionUpdates').click();
  await expect(page.locator('#subscriptionBadge')).toBeHidden();
  await expect(page.locator('.subscription-card .subscription-unread-count')).toHaveCount(0);
  await page.locator('#subscriptionVideoFilter').selectOption('unread');
  await expect(page.locator('.subscription-video')).toHaveCount(0);
  await expect(page.locator('.subscription-feed-empty')).toContainText('当前没有未读更新');
  await page.locator('#subscriptionVideoFilter').selectOption('pending');
  await expect(page.locator('.subscription-video')).toHaveCount(2);
  expect(errors).toEqual([]);
});

test('first-follow history stays out of inbox and mobile dialog can be cancelled', async ({ page }, testInfo) => {
  const { state, errors } = await fixture(page);
  state.videos = [state.videos[1]];
  state.unread_count = 0;
  await page.evaluate(() => window.MtdSubscriptions.refresh());
  await expect(page.locator('.subscription-feed-empty')).toContainText('暂时没有新投稿');
  await expect(page.locator('#subscriptionBadge')).toBeHidden();
  await page.locator('#subscriptionVideoFilter').selectOption('all');
  await expect(page.locator('.subscription-video-title')).toHaveText('历史投稿');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: testInfo.outputPath('subscriptions-mobile.png') });
  await page.locator('#addSubscription').click();
  await expect(page.locator('#subscriptionDialog')).toBeVisible();
  await page.locator('#subscriptionSource').fill('draft');
  await page.keyboard.press('Escape');
  await expect(page.locator('#subscriptionDialog')).not.toBeVisible();
  await page.locator('#addSubscription').click();
  await expect(page.locator('#subscriptionSource')).toHaveValue('');
  await page.locator('#subscriptionCancelEdit').click();
  await expect(page.locator('#subscriptionDialog')).not.toBeVisible();
  expect(state.creates).toEqual([]);
  expect(errors).toEqual([]);
});

test('opening subscriptions respects unsaved subtitle changes', async ({ page }) => {
  await fixture(page);
  await page.locator('.shell-nav[data-shell-page="tasks"]').click();
  await page.locator('.task-item[data-job-id="job-1"]').click();
  await expect(page.locator('#workbench')).toBeVisible();
  await page.locator('#segments tr[data-index="0"] textarea.text').fill('unsaved subtitle');
  page.once('dialog', (dialog) => dialog.dismiss());
  await page.locator('#editorSubscriptions').click();
  await expect(page.locator('#workbench')).toBeVisible();
  await expect(page.locator('#segments tr[data-index="0"] textarea.text')).toHaveValue('unsaved subtitle');
  page.once('dialog', (dialog) => dialog.accept());
  await page.locator('#editorSubscriptions').click();
  await expect(page.locator('#subscriptionsView')).toBeVisible();
});
