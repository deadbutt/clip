const { test, expect } = require('@playwright/test');
const { installApiFixture } = require('./fixtures');

/**
 * 让字幕浮窗在测试里可观测，同时不真的弹出窗口。
 *
 * 关掉 documentPictureInPicture 是刻意的：Chromium 上产品代码会优先走 Document PiP，
 * 那样 window.open 就不会被调用，测试也就看不到开窗行为。强制走弹窗分支后，
 * window.openedWindows 记录每次开窗请求（url + 窗口名），既能断言次数和顺序，
 * 也能断言直播间标签页用的是哪个命名窗口。
 */
async function stubSubtitleWindow(page) {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'documentPictureInPicture', { value: undefined, configurable: true });
    window.openedWindows = [];
    window.open = (url, name) => {
      window.openedWindows.push({ url, name });
      return { closed: false, focus() {}, close() {} };
    };
  });
}

test('live subtitles start independently from recording, render escaped bilingual output, and stop', async ({ page }, testInfo) => {
  await installApiFixture(page);
  await stubSubtitleWindow(page);
  let subtitles = { state: 'idle', rows: [] };
  let startPayload;
  await page.route('**/api/live/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    let body;
    if (path.endsWith('/platforms')) body = { platforms: ['哔哩哔哩', 'Twitch'] };
    else if (path.endsWith('/subtitles/start')) {
      startPayload = request.postDataJSON();
      subtitles = { state: 'listening', received_seconds: 12, processed_seconds: 12,
        rows: [{ id: 1, start: 0.5, end: 4.2, source: '<script>hello</script>',
          translation: '你好，世界。', translation_state: 'done' }] };
      body = subtitles;
    } else if (path.endsWith('/subtitles/stop')) {
      subtitles.state = 'stopped';
      body = subtitles;
    } else body = { sessions: [{ id: 'live-1', title: '测试直播', platform: '哔哩哔哩',
      url: 'https://live.bilibili.com/1', status: 'online', subtitles }] };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.goto('/');
  await page.locator('#openLive').click();
  await expect(page.locator('#liveView')).toBeVisible();
  await page.locator('[data-live-language]').selectOption('en');
  await page.getByRole('button', { name: '开启实时字幕' }).click();
  expect(startPayload).toEqual({ translate: true, language: 'en' });
  // 开启实时字幕本身就应该把字幕浮窗开出来，而不是只把内容写进卡片；
  // 这条路径不开直播间标签页，所以只应该有一次开窗。
  await expect.poll(() => page.evaluate(() => window.openedWindows.length)).toBe(1);
  await expect.poll(() => page.evaluate(() => (window.openedWindows[0] || {}).url || '')).toContain('/static/live-subtitles.html?session=live-1');
  await expect(page.locator('.live-subtitle-line .live-translation')).toHaveText('你好，世界。');
  await expect(page.locator('.live-subtitle-line')).toContainText('<script>hello</script>');
  await expect(page.locator('.live-subtitle-lines script')).toHaveCount(0);
  await expect(page.getByRole('button', { name: '开始录制', exact: true })).toBeEnabled();
  await page.screenshot({ path: testInfo.outputPath('live-bilingual.png'), fullPage: true });
  await page.getByRole('button', { name: '停止实时字幕' }).click();
  await expect(page.getByRole('button', { name: '开启实时字幕' })).toBeVisible();
  await expect(page.locator('.live-subtitles')).toContainText('已停止');
});

test('watch and translate opens the floating subtitle window and reuses the named stream tab', async ({ page }, testInfo) => {
  await installApiFixture(page);
  await stubSubtitleWindow(page);
  const posts = [];
  let subtitles = { state: 'idle', rows: [] };
  await page.route('**/api/live/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST') posts.push(path);
    let body;
    if (path.endsWith('/platforms')) body = { platforms: ['哔哩哔哩'] };
    else if (path.endsWith('/subtitles/start')) {
      subtitles = { state: 'listening', translate: true, rows: [{ id: 1, start: 0, source: 'Hello', translation: '你好' }] };
      body = subtitles;
    } else if (path.endsWith('/subtitles/stop')) { subtitles.state = 'stopped'; body = subtitles; }
    else body = { sessions: [{ id: 'live-1', title: '只看直播不录制', url: 'https://live.bilibili.com/1', status: 'online', subtitles }] };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.goto('/');
  await page.locator('#openLive').click();
  await page.getByRole('button', { name: '打开并实时翻译' }).click();
  await expect(page.locator('.live-subtitles .live-translation')).toHaveText('你好');
  // 这次点击要开两个窗：先字幕浮窗，再按固定名字复用/建立直播间标签页。
  // 复用已有命名窗口不占用“一次手势只能开一个窗”的名额，所以两者可以同时成立。
  await expect.poll(() => page.evaluate(() => window.openedWindows.length)).toBe(2);
  await expect.poll(() => page.evaluate(() => (window.openedWindows[0] || {}).url || '')).toContain('/static/live-subtitles.html?session=live-1');
  await expect.poll(() => page.evaluate(() => (window.openedWindows[1] || {}).name || '')).toBe('mtd-live-stream-live-1');
  await expect(page.getByRole('link', { name: '打开原直播间' })).toHaveAttribute('href', 'https://live.bilibili.com/1');
  // 链接必须用同一个命名 target，且不能带 noopener/noreferrer——它们会丢掉窗口名，复用就失效了。
  await expect(page.getByRole('link', { name: '打开原直播间' })).toHaveAttribute('target', 'mtd-live-stream-live-1');
  await expect(page.getByRole('link', { name: '打开原直播间' })).not.toHaveAttribute('rel', /noopener|noreferrer/);
  // 原生桌面浮窗的入口必须一直在卡片上（浏览器里的浮窗做不到桌面透明）。
  await expect(page.getByRole('button', { name: '桌面透明浮窗' })).toBeVisible();
  await page.locator('#liveRefresh').click();
  expect(posts.filter((path) => path.endsWith('/record'))).toEqual([]);
  expect(posts.filter((path) => path.endsWith('/preview/start'))).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('watch-translate.png'), fullPage: true });
  await page.getByRole('button', { name: '停止实时字幕' }).click();
  await expect(page.locator('.live-subtitles')).toContainText('已停止');
  expect(posts.filter((path) => path.endsWith('/record'))).toEqual([]);
});

test('desktop overlay button starts the native transparent window', async ({ page }) => {
  await installApiFixture(page);
  const posts = [];
  await page.route('**/api/live/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST') posts.push(path);
    let body;
    if (path.endsWith('/platforms')) body = { platforms: ['哔哩哔哩'] };
    else if (path.endsWith('/overlay/start')) body = { running: true, pid: 4242 };
    else if (path.endsWith('/overlay/stop')) body = { running: false };
    else body = { sessions: [{ id: 'live-1', title: '桌面浮窗', url: 'https://live.bilibili.com/1',
      status: 'online', subtitles: { state: 'listening', rows: [] } }] };
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
  });
  await page.goto('/');
  await page.locator('#openLive').click();
  await page.getByRole('button', { name: '桌面透明浮窗' }).click();
  await expect(page.locator('#liveMessage')).toContainText('桌面浮窗已启动');
  expect(posts.filter((path) => path.endsWith('/overlay/start')))
    .toEqual(['/api/live/sessions/live-1/overlay/start']);
});
