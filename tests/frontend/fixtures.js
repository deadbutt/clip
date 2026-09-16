const { expect } = require('@playwright/test');

function makeSegments(count = 4) {
  return Array.from({ length: count }, (_, index) => {
    const start = index * 2.5;
    return {
      id: `seg_${String(index + 1).padStart(4, '0')}`,
      start,
      end: start + 2,
      speaker: index % 2 ? 'S02' : 'S01',
      text: `subtitle ${index + 1} has words`,
      items: [
        { text: 'subtitle', start, end: start + 0.5 },
        { text: String(index + 1), start: start + 0.5, end: start + 1 },
        { text: 'has', start: start + 1, end: start + 1.5 },
        { text: 'words', start: start + 1.5, end: start + 2 },
      ],
    };
  });
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function makeJob() {
  return {
    id: 'job-1',
    status: 'waiting_review',
    progress: 0.95,
    media_name: 'fixture.mp4',
    updated_at: 100,
    backend: 'whisper',
    source: 'upload',
    inference: { prompt: '', max_length: 1024, max_new_tokens: 256, decoding: 'greedy' },
    usage: { generated_tokens: 4, max_new_tokens: 256 },
    speaker_labeling: { enabled: true, applied: true, speakers: 2, method: 'fixture' },
    translation: { applied: true, mode: 'bilingual', source_available: true, in_progress: false },
    proofread: { result_available: false },
    alignment: { result_available: false },
    subtitle_style: {},
  };
}

async function installApiFixture(page, options = {}) {
  const state = {
    job: makeJob(),
    segments: clone(options.segments || makeSegments(options.segmentCount || 4)),
    saveRequests: 0,
    splitRequests: [],
    mergeRequests: 0,
    clipRenderRequests: 0,
  };

  await page.addInitScript(() => {
    class FakeEventSource {
      constructor() { this.listeners = new Map(); }
      addEventListener(type, listener) { this.listeners.set(type, listener); }
      close() {}
    }
    window.EventSource = FakeEventSource;
    HTMLMediaElement.prototype.play = () => Promise.resolve();
    HTMLMediaElement.prototype.pause = () => {};
    HTMLMediaElement.prototype.load = () => {};
  });

  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();
    const json = (status, body, headers = {}) => route.fulfill({
      status,
      contentType: 'application/json',
      headers,
      body: JSON.stringify(body),
    });

    if (path === '/api/runtime') {
      return json(200, {
        ffmpeg: { available: true },
        translator: { available: true, model: 'fixture-opus' },
        inference: {}, speaker_labeling: {}, model: {},
      });
    }
    if (path === '/api/llm/profiles') return json(200, { profiles: [], active_id: null });
    if (path === '/api/jobs' && method === 'GET') return json(200, { jobs: [clone(state.job)] });
    if (path === '/api/jobs/job-1' && method === 'GET') return json(200, clone(state.job));
    if (path === '/api/jobs/job-1/media') return route.fulfill({ status: 200, contentType: 'video/mp4', body: '' });
    if (path === '/api/jobs/job-1/segments' && method === 'GET') {
      const etag = `"fixture-${state.saveRequests}"`;
      if (options.conditionalSegments && request.headers()['if-none-match'] === etag) {
        return route.fulfill({ status: 304, headers: { ETag: etag }, body: '' });
      }
      return json(200, { segments: clone(state.segments) }, { ETag: etag });
    }
    if (path === '/api/jobs/job-1/segments' && method === 'PUT') {
      state.saveRequests += 1;
      if (options.failSave) return json(500, { detail: '字幕保存失败：磁盘已满' });
      const payload = request.postDataJSON();
      state.segments = clone(payload.segments || payload);
      return json(200, { segments: clone(state.segments) });
    }
    if (path.endsWith('/split') && method === 'POST') {
      const payload = request.postDataJSON();
      state.splitRequests.push(clone(payload));
      const segmentId = decodeURIComponent(path.split('/').at(-2));
      const index = state.segments.findIndex((segment) => segment.id === segmentId);
      const segment = state.segments[index];
      const items = segment.items || [];
      let boundary = 1;
      if (items.length > 1) {
        boundary = items.slice(1).reduce((best, item, itemIndex) => (
          Math.abs(item.start - payload.time) < Math.abs(items[best].start - payload.time) ? itemIndex + 1 : best
        ), 1);
      }
      const leftItems = items.slice(0, boundary);
      const rightItems = items.slice(boundary);
      const sourceLeft = leftItems.map((item) => item.text).join(' ');
      const sourceRight = rightItems.map((item) => item.text).join(' ');
      const lines = String(segment.text).split('\n');
      let leftText = sourceLeft;
      let rightText = sourceRight;
      if (lines.length > 1) {
        const translated = lines[0];
        const ratio = payload.translation_ratio == null ? boundary / items.length : payload.translation_ratio;
        const cut = Math.max(1, Math.min(translated.length - 1, Math.round(translated.length * ratio)));
        leftText = `${translated.slice(0, cut).trimEnd()}\n${sourceLeft}`;
        rightText = `${translated.slice(cut).trimStart()}\n${sourceRight}`;
      }
      const splitTime = rightItems[0]?.start ?? payload.time;
      state.segments.splice(index, 1,
        { ...segment, end: splitTime, text: leftText, items: leftItems },
        { ...segment, id: `${segment.id}~2`, start: splitTime, text: rightText, items: rightItems });
      return json(200, { segments: clone(state.segments), needs_retranslate: lines.length > 1 });
    }
    if (path.endsWith('/segments/merge') && method === 'POST') {
      state.mergeRequests += 1;
      const ids = request.postDataJSON().ids;
      const firstIndex = state.segments.findIndex((segment) => segment.id === ids[0]);
      const group = state.segments.filter((segment) => ids.includes(segment.id));
      const first = group[0];
      const merged = {
        ...first,
        end: group.at(-1).end,
        text: group.map((segment) => segment.text).join(' '),
        items: group.flatMap((segment) => segment.items || []),
      };
      state.segments.splice(firstIndex, group.length, merged);
      return json(200, { segments: clone(state.segments), needs_retranslate: false });
    }
    if (path === '/api/jobs/job-1/clips' && method === 'GET') {
      return json(200, { clips: [
        { id: 'c1', start: 0, end: 3, title: 'one', score: 90, lane: 'primary' },
        { id: 'c2', start: 3, end: 6, title: 'two', score: 80, lane: 'primary' },
        { id: 'c3', start: 6, end: 9, title: 'three', score: 70, lane: 'primary' },
      ] });
    }
    if (path === '/api/jobs/job-1/clips/render' && method === 'POST') {
      state.clipRenderRequests += 1;
      if (options.failFirstClip && state.clipRenderRequests === 1) return json(500, { detail: '媒体处理失败' });
      return json(200, { filename: `clip-${state.clipRenderRequests}.mp4`, start: 0, end: 3 });
    }
    if (path.endsWith('/proofread') || path.endsWith('/alignment')) return json(404, { detail: '暂无结果' });
    return json(200, {});
  });
  return state;
}

async function openEditor(page) {
  await page.goto('/');
  const task = page.locator('.task-item[data-job-id="job-1"]');
  await expect(task).toBeVisible();
  await task.click();
  await expect(page.locator('#workbench')).toBeVisible();
  await expect(page.locator('#segments tr[data-index="0"]')).toBeVisible();
}

module.exports = { installApiFixture, makeSegments, openEditor };
