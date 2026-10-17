(() => {
  'use strict';
  const { apiUrl, escapeHtml: esc } = window.MtdWorkbenchUtils;
  const $ = (id) => document.getElementById(id);
  let busy = false;
  let refreshing = false;
  const preferences = new Map();
  let subtitleSessionId = null;
  // 字幕浮窗。在 Chromium 上是 Document Picture-in-Picture 窗口（系统级置顶、可拖动缩放），
  // 在 Firefox/Safari 上是普通弹窗（可拖动，但不置顶）。两者都放进 subtitleWindow。
  let subtitleWindow = null;
  let subtitleWindowKind = null;
  let subtitleWindowSessionId = null;
  let latestSessions = [];

  const labels = {
    idle: '未检查', online: '正在直播', offline: '未开播', unavailable: '引擎未安装',
    recording: '录制中', stopping: '正在保存', completed: '录制已保存', error: '操作失败',
  };

  // 直播间标签页固定使用同一个窗口名。命名窗口一旦开过，之后 window.open(url, 同名)
  // 就是“复用并切到前台”，实测这不占用“一次手势只能开一个窗”的名额，所以字幕浮窗
  // 和直播间可以在同一次点击里都拿到。
  const streamWindowName = (id) => `mtd-live-stream-${id}`;

  function focusStreamPage(id, url) {
    if (!url) return false;
    try { return !!window.open(url, streamWindowName(id)); } catch (_) { return false; }
  }

  async function request(path, method = 'GET', payload) {
    const response = await fetch(apiUrl(path), {
      method,
      cache: 'no-store',
      ...(payload ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } : {}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '直播操作失败');
    return data;
  }

  function message(text, error = false) {
    $('liveMessage').textContent = text;
    $('liveMessage').className = error ? 'error' : 'meta';
  }

  function subtitleHtml(item) {
    const sub = item.subtitles || { state: 'idle', rows: [] };
    const running = !['idle', 'stopped', 'error'].includes(sub.state);
    // 中文直播是本项目最常见的场景；锁定语言可避免游戏音效/短窗口误判成俄语等语言。
    const settings = preferences.get(item.id) || { translate: true, language: 'zh' };
    const subLabels = {
      idle: '未开启', starting: '正在启动', recognizing: '正在识别（或等待模型）',
      listening: '等待下一段音频', stopping: '正在停止', stopped: '已停止', error: '识别失败',
    };
    const translationLabels = { pending: '待翻译', skipped: '积压跳过', failed: '翻译失败', cancelled: '已取消' };
    const rows = (sub.rows || []).slice(-30).map((row) => `
      <div class="live-subtitle-line">
        <time>${Number(row.start).toFixed(1)}s</time>
        <div><div>${esc(row.source)}</div>${row.translation
          ? `<div class="live-translation">${esc(row.translation)}</div>`
          : `<div class="meta">${esc(translationLabels[row.translation_state] || '')}</div>`}
          ${row.translation_error ? `<div class="error">${esc(row.translation_error)}</div>` : ''}
        </div>
      </div>`).join('');
    return `<section class="live-subtitles" aria-label="实时字幕">
      <div class="live-subtitle-heading"><h4>字幕记录（悬浮窗同步）</h4><span class="meta">原文 / 中文翻译</span></div>
      <div class="live-subtitle-controls">
        <label>识别语言 <select data-live-language="${item.id}" ${running ? 'disabled' : ''}>${[
          ['zh', '中文（推荐）'], ['', '自动识别'], ['en', '英语'], ['ja', '日语'], ['ko', '韩语'],
        ].map(([value, label]) => `<option value="${value}" ${settings.language === value ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
        <label><input type="checkbox" data-live-translate="${item.id}" ${settings.translate ? 'checked' : ''} ${running ? 'disabled' : ''}> Index 实时译成中文</label>
        <button class="ghost small" data-live-action="${running ? 'subtitles/stop' : 'subtitles/start'}" data-id="${item.id}" ${sub.state === 'stopping' ? 'disabled' : ''}>${running ? '停止实时字幕' : '开启实时字幕'}</button>
        ${sub.draft_path ? `<a class="ghost small" href="${esc(apiUrl(`/api/live/sessions/${item.id}/subtitles/draft`))}" download>下载双语草稿</a>` : ''}
      </div>
      <div class="meta" role="status">${esc(subLabels[sub.state] || sub.state)}${running
        ? ` · 已接收 ${Number(sub.received_seconds || 0).toFixed(0)} 秒 · 识别积压 ${Number(sub.backlog_seconds || 0).toFixed(1)} 秒 · 翻译排队 ${Number(sub.translation_queue || 0)} 段`
        : ''}${sub.dropped_windows ? ` · 跳过 ${Number(sub.dropped_windows)} 个音频窗口` : ''}</div>
      ${sub.error ? `<div class="error">${esc(sub.error)}</div>` : ''}${sub.warning ? `<div class="meta">${esc(sub.warning)}</div>` : ''}
      <div class="live-subtitle-lines" data-live-scroll="${item.id}">${rows || '<p class="meta">点“打开并实时翻译”会弹出可拖动的置顶字幕浮窗（Chrome / Edge 为 PiP 浮窗，其它浏览器为独立弹窗）；这里保留滚动记录。</p>'}</div>
    </section>`;
  }

  async function refresh() {
    if (refreshing) return;
    if ($('liveSessions').contains(document.activeElement) && document.activeElement.matches('select')) return;
    refreshing = true;
    try {
      const data = await request('/api/live/sessions');
      latestSessions = data.sessions;
      const scrollPositions = new Map([...$('liveSessions').querySelectorAll('[data-live-scroll]')].map((element) => [
        element.dataset.liveScroll,
        { top: element.scrollTop, bottom: element.scrollHeight - element.clientHeight - element.scrollTop < 30 },
      ]));
      $('liveSessions').innerHTML = data.sessions.length ? data.sessions.map((item) => {
        const recording = ['recording', 'stopping'].includes(item.status);
        const sub = item.subtitles || { state: 'idle', rows: [] };
        const running = !['idle', 'stopped', 'error'].includes(sub.state);
        return `<article class="panel live-card">
          <div class="live-card-head"><div><h3>${esc(item.title || item.platform + '直播间')}</h3><span class="live-url">${esc(item.url)}</span></div><span class="live-status ${esc(item.status)}">${esc(labels[item.status] || item.status)}</span></div>
          <div class="live-card-body"><div class="live-card-main">
            ${item.error ? `<div class="error">${esc(item.error)}</div>` : ''}
            ${item.recording_path ? `<div class="live-file">录制文件：${esc(item.recording_path)}</div>` : ''}
            <div class="live-card-actions"><!-- 不能加 rel="noopener"/"noreferrer"：它们会让新窗口丢掉 target 名，
  之后就无法按名字复用（实测）。-->${''}<a class="ghost small" href="${esc(item.url)}" target="${esc(streamWindowName(item.id))}">打开原直播间</a>
              <button class="primary small" data-live-action="watch-translate" data-id="${item.id}">打开并实时翻译</button>
              <button class="ghost small" data-live-action="overlay/start" data-id="${item.id}">桌面透明浮窗</button>
              <button class="ghost small" data-live-action="watch" data-id="${item.id}">只观看（原站）</button>
              <button class="ghost small" data-live-action="resolve" data-id="${item.id}" ${recording ? 'disabled' : ''}>检查直播</button>
              <button class="ghost small" data-live-action="${recording ? 'stop' : 'record'}" data-id="${item.id}" ${item.status === 'stopping' ? 'disabled' : ''}>${recording ? '停止录制' : '开始录制'}</button>
              ${item.recording_path && !recording ? `<button class="ghost small" data-live-action="transcribe" data-id="${item.id}">转录录制视频</button>` : ''}
              <button class="ghost small" data-live-action="delete" data-id="${item.id}" ${recording || running ? 'disabled' : ''}>移除直播间</button>
            </div>
            <p class="meta live-watch-hint">“打开并实时翻译”弹出可拖动的字幕浮窗（Chrome / Edge 为置顶浮窗，可以压在全屏游戏上），并同时把已开过的直播间标签页切到前台；直播画面第一次请用左侧“打开原直播间”，之后就会自动跟随。</p>
          </div>${subtitleHtml(item)}</div>
        </article>`;
      }).join('') : '<div class="panel live-empty"><h3>还没有直播间</h3><p class="meta">添加主播的直播链接，从观看或录制开始。</p></div>';
      $('liveSessions').querySelectorAll('[data-live-scroll]').forEach((element) => {
        const previous = scrollPositions.get(element.dataset.liveScroll);
        element.scrollTop = !previous || previous.bottom ? element.scrollHeight : previous.top;
      });
    } finally { refreshing = false; }
  }

  $('liveSessions').addEventListener('change', (event) => {
    const id = event.target.dataset.liveLanguage || event.target.dataset.liveTranslate;
    if (!id) return;
    const settings = preferences.get(id) || { translate: true, language: 'zh' };
    if (event.target.dataset.liveLanguage) settings.language = event.target.value;
    else settings.translate = event.target.checked;
    preferences.set(id, settings);
  });

  $('openLive').addEventListener('click', async () => {
    if (!confirmLeaveUnsavedChanges()) return;
    stopPreviewPlayback();
    setVisible($('liveView'));
    try {
      const data = await request('/api/live/platforms');
      $('livePlatforms').textContent = '平台适配：' + data.platforms.join(' · ');
      await refresh();
    } catch (error) { message(error.message, true); }
  });

  function subtitleWindowAlive() {
    if (!subtitleWindow) return false;
    try { return !subtitleWindow.closed; } catch (_) { return false; }
  }

  function closeSubtitleWindow() {
    const win = subtitleWindow;
    subtitleWindow = null;
    subtitleWindowKind = null;
    subtitleWindowSessionId = null;
    if (!win) return;
    try { if (!win.closed) win.close(); } catch (_) { /* PiP 窗口可能已被用户关掉 */ }
  }

  /** 打开字幕浮窗。
   *
   * Chromium（Chrome / Edge 116+）走 Document Picture-in-Picture：拿到的是系统级置顶、
   * 可以用鼠标拖动和缩放的浮窗，能压在全屏游戏和其它应用上面。Firefox / Safari 没有这个
   * API，自动退回普通弹窗——同样能拖动，只是不会置顶（网页端没有让窗口置顶的标准手段）。
   *
   * 注意：浏览器一次用户手势只放行一次开窗调用，所以本函数必须在点击处理器的任何 await
   * 之前被调用，并且是同一次点击里唯一一次开窗。
   */
  async function openSubtitleWindow(id) {
    const url = `${apiUrl('/static/live-subtitles.html')}?session=${encodeURIComponent(id)}`;
    if (subtitleWindowAlive()) {
      if (subtitleWindowSessionId === id) {
        try { subtitleWindow.focus(); } catch (_) {}
        return subtitleWindow;
      }
      closeSubtitleWindow();
    }
    subtitleWindowSessionId = id;

    if (window.documentPictureInPicture) {
      try {
        const pip = await window.documentPictureInPicture.requestWindow({ width: 460, height: 320 });
        const frame = pip.document.createElement('iframe');
        frame.src = url;
        frame.title = '实时字幕';
        frame.style.cssText = 'display:block;width:100%;height:100%;border:0';
        pip.document.body.style.cssText = 'margin:0;height:100vh;overflow:hidden;background:#17252a';
        pip.document.body.append(frame);
        pip.addEventListener('pagehide', () => {
          if (subtitleWindow === pip) {
            subtitleWindow = null;
            subtitleWindowKind = null;
            subtitleWindowSessionId = null;
          }
        });
        subtitleWindow = pip;
        subtitleWindowKind = 'pip';
        return pip;
      } catch (error) {
        // 用户拒绝了浮窗，或本浏览器实现有问题——继续往下走弹窗兜底。
        console.warn('Document PiP 不可用，退回普通弹窗', error);
      }
    }

    const popup = window.open(url, `mtd-live-subtitles-${id}`, 'popup=yes,width=460,height=320,resizable=yes,scrollbars=yes');
    if (!popup) {
      subtitleWindow = null;
      subtitleWindowKind = null;
      subtitleWindowSessionId = null;
      return null;
    }
    subtitleWindow = popup;
    subtitleWindowKind = 'popup';
    return popup;
  }

  // PiP 里承载的是 live-subtitles.html 这个 iframe，它点“停止并关闭”时关不掉外层浮窗，
  // 因此由它 postMessage 回来，这里负责收尾。
  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin) return;
    if ((event.data || {}).type === 'mtd-live-subtitles-closed') closeSubtitleWindow();
  });

  window.stopLiveViewer = () => {
    if (!subtitleSessionId) return;
    const id = subtitleSessionId;
    subtitleSessionId = null;
    closeSubtitleWindow();
    fetch(apiUrl(`/api/live/sessions/${id}/subtitles/stop`), { method: 'POST', keepalive: true }).catch(() => {});
  };

  $('liveRefresh').addEventListener('click', () => refresh().catch((error) => message(error.message, true)));
  $('liveAdd').addEventListener('click', async () => {
    if (busy) return;
    busy = true; $('liveAdd').disabled = true; message('正在添加并检查直播…');
    try {
      await request('/api/live/sessions', 'POST', { url: $('liveUrl').value, quality: $('liveQuality').value });
      $('liveUrl').value = ''; message('直播间已添加'); await refresh();
    } catch (error) { message(error.message, true); }
    finally { busy = false; $('liveAdd').disabled = false; }
  });

  $('liveSessions').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-live-action]');
    if (!button || busy) return;
    busy = true; button.disabled = true;
    const { id, liveAction: action } = button.dataset;
    try {
      if (action === 'watch' || action === 'watch-translate') {
        const current = latestSessions.find((item) => item.id === id);
        if (action === 'watch') {
          // 用命名窗口打开，这样后续“打开并实时翻译”可以复用它、把它切到前台。
          const opened = focusStreamPage(id, current && current.url);
          message(opened ? '已打开原直播间；需要字幕时点击“打开并实时翻译”'
                         : '浏览器拦截了直播间标签页，请允许本站弹窗后重试', !opened);
          await refresh();
          return;
        }
        // 一次手势只放行一次“新建窗口”，但复用已有命名窗口不受这条限制（实测）。
        // 所以先拿字幕浮窗，再把之前开过的直播间标签页切到前台；两者可以同时成立。
        const popup = await openSubtitleWindow(id);
        const streamFocused = focusStreamPage(id, current && current.url);
        if (subtitleSessionId && subtitleSessionId !== id) {
          await request(`/api/live/sessions/${subtitleSessionId}/subtitles/stop`, 'POST');
        }
        subtitleSessionId = id;
        const state = ((current || {}).subtitles || {}).state || 'idle';
        if (['idle', 'stopped', 'error'].includes(state)) {
          const settings = { ...(preferences.get(id) || { language: 'zh' }), translate: true };
          preferences.set(id, settings);
          await request(`/api/live/sessions/${id}/subtitles/start`, 'POST', settings);
        } else if (current.subtitles.translate === false) {
          message('当前字幕为仅原文模式；停止实时字幕后重新打开可开启翻译');
        }
        message(popup
          ? (streamFocused
            ? '字幕浮窗已打开，直播间标签页已切到前台'
            : '字幕浮窗已打开；直播画面先点一次“打开原直播间”，之后这里会自动带上它')
          : '字幕浮窗被浏览器拦截：请允许本站弹窗后重试', !popup);
        await refresh();
        return;
      }

      if (action === 'subtitles/start') {
        // 开窗必须在任何 await 之前，且是本次点击里唯一一次开窗。
        const popup = await openSubtitleWindow(id);
        await request(`/api/live/sessions/${id}/subtitles/start`, 'POST', preferences.get(id) || { translate: true, language: 'zh' });
        subtitleSessionId = id;
        message(popup ? '已开启实时字幕，浮窗已打开' : '字幕浮窗被浏览器拦截：请允许本站弹窗后重试', !popup);
        await refresh();
        return;
      }

      if (action === 'overlay/start') {
        // 原生浮窗：由服务端另起一个进程，不占用浏览器的开窗名额。
        const result = await request(`/api/live/sessions/${id}/overlay/start`, 'POST');
        message(result && result.running
          ? '桌面浮窗已启动：可拖动、右键调字号或关闭，透明区域鼠标会穿透'
          : '桌面浮窗启动失败：请确认 Python 环境带 tkinter');
        await refresh();
        return;
      }

      if (action === 'overlay/stop') {
        await request(`/api/live/sessions/${id}/overlay/stop`, 'POST');
        message('桌面浮窗已关闭');
        await refresh();
        return;
      }

      if (action === 'subtitles/stop') closeSubtitleWindow();

      const result = action === 'transcribe'
        ? await request(`/api/live/recordings/${id}/transcribe`, 'POST')
        : await request(`/api/live/sessions/${id}${action === 'delete' ? '' : '/' + action}`, action === 'delete' ? 'DELETE' : 'POST');
      message(action === 'transcribe' ? `转录任务已创建：${result.id}` : '操作完成');
      await refresh();
    } catch (error) { message(error.message, true); }
    finally { busy = false; button.disabled = false; }
  });

  setInterval(() => {
    if (!$('liveView').classList.contains('is-hidden') && !busy) refresh().catch(() => {});
  }, 3000);
})();
