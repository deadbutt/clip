const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  await page.goto('http://127.0.0.1:7864/', { waitUntil: 'networkidle' });

  const result = await page.evaluate(() => {
    currentJob = { id: 'test', media_name: 'test.mp4', status: 'edit', segments: [
      { id: 's1', start: 0, end: 2, speaker: 'S01', text: 'first' },
      { id: 's2', start: 3, end: 5, speaker: 'S02', text: 'second' }
    ]};
    setVisible(workbench);
    renderSegments(currentJob.segments);

    const segs = timelineLane.querySelectorAll('.timeline-segment');
    const guide = document.querySelector('#timelineGuide');
    if (segs.length < 2) return { error: 'no segments', count: segs.length };

    const seg = segs[1];
    const rect = seg.getBoundingClientRect();
    const startX = Math.round(rect.left + rect.width / 2);
    const startY = Math.round(rect.top + rect.height / 2);
    const pps = currentPixelsPerSecond || 12;

    seg.dispatchEvent(new PointerEvent('pointerdown', { clientX: startX, clientY: startY, pointerId: 1, button: 0, bubbles: true }));
    // 向左拖 1s: seg2 start 3 -> 2, 靠近 seg1 end(2), 应触发 snap
    const moveX = startX - Math.round(pps * 1);
    window.dispatchEvent(new PointerEvent('pointermove', { clientX: moveX, clientY: startY, pointerId: 1, bubbles: true }));

    const after = {
      segCount: segs.length,
      pps,
      guideExists: !!guide,
      guideVisible: guide && guide.classList.contains('visible'),
      guideSnapped: guide && guide.classList.contains('snapped'),
      guideLeft: guide && guide.style.left,
      guideDisplay: guide && getComputedStyle(guide).display,
      guideBorder: guide && getComputedStyle(guide).borderLeftColor,
      seg2Left: seg.style.left,
      seg2Width: seg.style.width,
      dragMoved: segmentDragState && segmentDragState.moved
    };
    // 再拖到一个不靠近任何段的位置,看 snapped 是否取消
    const moveX2 = startX - Math.round(pps * 0.3);
    window.dispatchEvent(new PointerEvent('pointermove', { clientX: moveX2, clientY: startY, pointerId: 1, bubbles: true }));
    after.guideSnappedAfter = guide && guide.classList.contains('snapped');
    after.guideVisibleAfter = guide && guide.classList.contains('visible');
    return after;
  });

  console.log(JSON.stringify(result, null, 2));
  const tlPanel = await page.$('.timeline-panel');
  if (tlPanel) {
    await tlPanel.screenshot({ path: '_guide_snap.png' });
    console.log('screenshot saved: _guide_snap.png');
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
