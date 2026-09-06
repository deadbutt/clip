(function exposeWorkbenchUtilities(global) {
  function apiUrl(path) {
    const clean = String(path).replace(/^[/]+/, '');
    const basePath = global.location.pathname.endsWith('/') ? global.location.pathname : global.location.pathname + '/';
    return new URL(clean, global.location.origin + basePath).toString();
  }

  function normalizedSpeakerCount(value) {
    if (value === '' || value == null) return '';
    const count = Number(value);
    if (!Number.isFinite(count) || count <= 0) return '';
    return String(Math.max(1, Math.min(10, Math.round(count))));
  }

  function formatTimelineTime(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    const total = Math.floor(seconds);
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }

  function formatDuration(seconds) {
    seconds = Math.max(0, Math.round(Number(seconds) || 0));
    const minutes = Math.floor(seconds / 60);
    const rest = seconds % 60;
    if (minutes <= 0) return rest + 's';
    return minutes + 'm ' + String(rest).padStart(2, '0') + 's';
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function encodeClipPayload(clip) {
    try {
      return escapeHtml(JSON.stringify(clip));
    } catch (_) {
      return '';
    }
  }

  function decodeClipPayload(value) {
    try {
      return JSON.parse(value || '{}');
    } catch (_) {
      return null;
    }
  }

  function makeClipId() {
    return 'clip_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 7);
  }

  function normalizeQueuedClip(source) {
    const start = Math.max(0, Number(source.start) || 0);
    const fallbackEnd = start + Math.max(0.25, Number(source.duration) || 120);
    const end = Math.max(start + 0.25, Number(source.end) || fallbackEnd);
    return {
      id: makeClipId(),
      sourceId: source.id || '',
      start,
      end,
      title: String(source.title || '未命名片段').trim() || '未命名片段',
      reason: String(source.reason || ''),
      score: Number(source.score) || 0,
      selectionMethod: source.selection_method || source.selectionMethod || 'rules',
      lane: source.lane === 'alternate' ? 'alternate' : 'primary',
      parentId: source.parent_id || source.parentId || ''
    };
  }

  global.MtdWorkbenchUtils = {
    apiUrl,
    normalizedSpeakerCount,
    formatTimelineTime,
    formatDuration,
    escapeHtml,
    encodeClipPayload,
    decodeClipPayload,
    makeClipId,
    normalizeQueuedClip
  };
})(window);
