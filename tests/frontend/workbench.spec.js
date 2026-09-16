const { test, expect } = require('@playwright/test');
const { installApiFixture, makeSegments, openEditor } = require('./fixtures');

function captureJsErrors(page) {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('Failed to load resource')) errors.push(message.text());
  });
  return errors;
}

test('page loads without JavaScript errors and task list opens the editor', async ({ page }) => {
  const errors = captureJsErrors(page);
  await installApiFixture(page);
  await openEditor(page);
  await expect(page.locator('#selectedName')).toHaveText('fixture.mp4');
  await expect(page.locator('#segments tr[data-index]')).toHaveCount(4);
  expect(errors).toEqual([]);
});

test('edited subtitle survives save and page reload', async ({ page }) => {
  const state = await installApiFixture(page);
  await openEditor(page);
  const text = page.locator('#segments tr[data-index="0"] textarea.text');
  await text.fill('edited subtitle survives reload');
  await expect(page.locator('#saveStatus')).toHaveText('有未保存修改');
  await page.locator('#save').click();
  await expect(page.locator('#saveStatus')).toHaveText('已保存');
  expect(state.saveRequests).toBe(1);

  await page.reload();
  await page.locator('.task-item[data-job-id="job-1"]').click();
  await expect(page.locator('#segments tr[data-index="0"] textarea.text')).toHaveValue('edited subtitle survives reload');
});

test('split, merge, undo and redo preserve editor structure', async ({ page }) => {
  const state = await installApiFixture(page);
  await openEditor(page);
  const first = page.locator('#segments tr[data-index="0"] textarea.text');
  await first.focus();
  await first.evaluate((element) => element.setSelectionRange(11, 11));
  await first.press('Control+Enter');
  await expect.poll(() => state.segments.length).toBe(5);
  await expect(page.locator('#segments tr[data-index]')).toHaveCount(5);

  await page.locator('#segments tr[data-index="0"] .merge-row-below').click();
  await expect.poll(() => state.segments.length).toBe(4);
  await expect(page.locator('#segments tr[data-index]')).toHaveCount(4);

  await page.locator('#undoBtn').click();
  await expect.poll(() => state.segments.length).toBe(5);
  await page.locator('body').press('Control+y');
  await expect.poll(() => state.segments.length).toBe(4);
  expect(state.mergeRequests).toBe(1);
});

test('Chinese split maps to source word order and keeps both languages', async ({ page }) => {
  const segments = [{
    id: 'seg_0001', start: 0, end: 10, speaker: 'S01', text: '甲乙丙丁 戊\none two three four five',
    items: [
      { text: 'one', start: 0, end: 0.5 },
      { text: 'two', start: 5, end: 6 },
      { text: 'three', start: 6, end: 7 },
      { text: 'four', start: 7, end: 8 },
      { text: 'five', start: 8, end: 10 },
    ],
    bilingual_chunks: [
      { translation: '甲乙丙丁', source: 'one', start: 0, end: 0.5, item_count: 1 },
      { translation: '戊', source: 'two three four five', start: 5, end: 10, item_count: 4 },
    ],
  }];
  const state = await installApiFixture(page, { segments });
  await openEditor(page);
  const text = page.locator('#segments tr[data-index="0"] textarea.text');
  await text.focus();
  await text.evaluate((element) => element.setSelectionRange(4, 4));
  await text.press('Control+Enter');
  await expect.poll(() => state.splitRequests.length).toBe(1);
  expect(state.splitRequests[0].time).toBe(5);
  expect(state.segments.map((segment) => segment.text)).toEqual([
    '甲乙丙丁\none',
    '戊\ntwo three four five',
  ]);
});

test('Chinese split skips untranslated source chunks in a mixed segment', async ({ page }) => {
  const segments = [{
    id: 'seg_0001', start: 0, end: 4, speaker: 'S01', text: '你好\nhello friend world again',
    items: [
      { text: 'hello', start: 0, end: 1 },
      { text: 'friend', start: 1, end: 2 },
      { text: 'world', start: 2, end: 3 },
      { text: 'again', start: 3, end: 4 },
    ],
    bilingual_chunks: [
      { translation: '', source: 'hello friend', start: 0, end: 2, item_count: 2 },
      { translation: '你好', source: 'world again', start: 2, end: 4, item_count: 2 },
    ],
  }];
  const state = await installApiFixture(page, { segments });
  await openEditor(page);
  const text = page.locator('#segments tr[data-index="0"] textarea.text');
  await text.focus();
  await text.evaluate((element) => element.setSelectionRange(0, 0));
  await text.press('Control+Enter');
  await expect.poll(() => state.splitRequests.length).toBe(1);
  expect(state.splitRequests[0].time).toBe(2);
  expect(state.splitRequests[0].translation_ratio).toBe(0);
});

test('timeline drag snaps and playback activates the matching subtitle', async ({ page }) => {
  await installApiFixture(page);
  await openEditor(page);
  const playbackState = await page.locator('#preview').evaluate((video) => {
    Object.defineProperty(video, 'duration', { configurable: true, value: 20 });
    Object.defineProperty(video, 'currentTime', { configurable: true, writable: true, value: 3 });
    video.dispatchEvent(new Event('timeupdate'));
    syncActiveSegment(true);
    return {
      time: video.currentTime,
      match: findSegmentIndexAtTime(collectSegments(), video.currentTime),
      activeRows: document.querySelectorAll('#segments tr.active').length,
    };
  });
  expect(playbackState).toEqual({ time: 3, match: 1, activeRows: 1 });
  await expect(page.locator('#segments tr[data-index="1"]')).toHaveClass(/active/);
  await expect(page.locator('#subtitleOverlay')).toContainText('subtitle 2');

  const snapped = await page.evaluate(() => {
    const segment = document.querySelectorAll('.timeline-segment')[1];
    const rect = segment.getBoundingClientRect();
    const startX = rect.left + 1;
    const startY = rect.top + rect.height / 2;
    segment.dispatchEvent(new PointerEvent('pointerdown', {
      clientX: startX, clientY: startY, pointerId: 1, button: 0, bubbles: true,
    }));
    window.dispatchEvent(new PointerEvent('pointermove', {
      clientX: startX - currentPixelsPerSecond * 0.5 * 5,
      clientY: startY, pointerId: 1, bubbles: true,
    }));
    const result = document.querySelector('#timelineGuide').classList.contains('snapped');
    window.dispatchEvent(new PointerEvent('pointercancel', { pointerId: 1, bubbles: true }));
    return result;
  });
  expect(snapped).toBe(true);
});

test('virtual list can select the final subtitle in a long task', async ({ page }) => {
  await installApiFixture(page, { segments: makeSegments(240) });
  await openEditor(page);
  const table = page.locator('.table-wrap');
  await table.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
    element.dispatchEvent(new Event('scroll'));
  });
  const last = page.locator('#segments tr[data-index="239"]');
  await expect(last).toBeVisible();
  await last.locator('.speaker').click();
  await expect(last).toHaveClass(/active/);
  const currentTime = await page.locator('#preview').evaluate((video) => video.currentTime);
  expect(currentTime).toBeCloseTo(239 * 2.5, 1);
});

test('returning from a late timeline position reloads the full virtual list', async ({ page }) => {
  await installApiFixture(page, { segmentCount: 240, conditionalSegments: true });
  await openEditor(page);

  await page.evaluate(() => {
    timelineScroll.scrollLeft = 200 * 2.5 * currentPixelsPerSecond;
    renderVisibleTimelineSegments();
  });
  await page.locator('.timeline-segment[data-index="200"]').click();
  await expect(page.locator('#segments tr[data-index="200"]')).toBeVisible();

  await page.locator('#backToTasks').click();
  await page.locator('.task-item[data-job-id="job-1"]').click();
  await expect(page.locator('#workbench')).toBeVisible();

  await page.locator('.table-wrap').evaluate((element) => {
    element.scrollTop = 0;
    element.dispatchEvent(new Event('scroll'));
  });
  await expect(page.locator('#segments tr[data-index="0"]')).toBeVisible();
  expect(await page.evaluate(() => cachedSegments?.length)).toBe(240);
});

test('unsaved changes require confirmation before leaving the editor', async ({ page }) => {
  await installApiFixture(page);
  await openEditor(page);
  await page.locator('#segments tr[data-index="0"] textarea.text').fill('unsaved change');

  page.once('dialog', async (dialog) => {
    expect(dialog.message()).toContain('未保存');
    await dialog.dismiss();
  });
  await page.locator('#backToTasks').click();
  await expect(page.locator('#workbench')).toBeVisible();

  page.once('dialog', (dialog) => dialog.accept());
  await page.locator('#backToTasks').click();
  await expect(page.locator('#importView')).toBeVisible();
});

test('batch clips continue after one item fails', async ({ page }) => {
  const state = await installApiFixture(page, { failFirstClip: true });
  await openEditor(page);
  await page.locator('#openClips').click();
  await page.locator('#findClipsRules').click();
  await page.locator('#startClipAnalysis').click();
  await expect(page.locator('#clipList .clip-card')).toHaveCount(3);
  await page.locator('#renderClipQueue').click();
  await expect.poll(() => state.clipRenderRequests).toBe(3);
  await expect(page.locator('#clipStatus')).toContainText('2 个成功');
  await expect(page.locator('#clipStatus')).toContainText('1 个失败');
});

test('settings, translation and proofreading modals have working state transitions', async ({ page }) => {
  await installApiFixture(page);
  await openEditor(page);
  for (const [open, modal, close] of [
    ['#openSettings', '#settingsModal', '#closeSettings'],
    ['#openTranslate', '#translateModal', '#closeTranslate'],
    ['#openProofread', '#proofreadModal', '#closeProofread'],
  ]) {
    await page.locator(open).click();
    await expect(page.locator(modal)).toBeVisible();
    await page.locator(close).click();
    await expect(page.locator(modal)).toBeHidden();
  }
});

test('backend save errors are shown as a readable message', async ({ page }) => {
  await installApiFixture(page, { failSave: true });
  await openEditor(page);
  await page.locator('#segments tr[data-index="0"] textarea.text').fill('cannot save this');
  await page.locator('#save').click();
  await expect(page.locator('#taskNotice')).toContainText('字幕保存失败：磁盘已满');
  await expect(page.locator('#saveStatus')).toContainText('磁盘已满');
});
