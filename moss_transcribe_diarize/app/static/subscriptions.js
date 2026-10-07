(() => {
  'use strict';
  const { apiUrl, escapeHtml } = window.MtdWorkbenchUtils;
  const $ = (id) => document.getElementById(id);
  const view = $('subscriptionsView');
  const form = $('subscriptionForm');
  let data = { subscriptions: [], videos: [], unread_count: 0 };
  let editingId = null;
  let selectedId = null;
  let loading = false;
  let busy = false;
  const noticeKey = 'mtd-subscription-notifications';
  const notifiedKey = 'mtd-subscription-notified-ids';
  const date = (value) => value ? new Date(value * 1000).toLocaleString('zh-CN', { hour12: false }) : '尚未检查';
  const platformName = (value) => value === 'youtube' ? 'YouTube' : 'B 站';
  const displayName = (sub) => sub.name || sub.channel_name || sub.channel_id;
  const unreadFor = (id) => data.videos.filter((v) => v.subscription_id === id && v.unread).length;

  function message(text, error = false) {
    $('subscriptionsMessage').textContent = text;
    $('subscriptionsMessage').className = error ? 'error' : 'meta';
    $('subscriptionFormMessage').textContent = text;
    $('subscriptionFormMessage').className = error ? 'error' : 'meta';
  }
  async function request(path, method = 'GET', payload) {
    const response = await fetch(apiUrl(path), {
      method, cache: 'no-store',
      ...(payload === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '操作失败，请重试');
    return result;
  }
  function storageGet(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* Badge remains available without storage. */ }
  }
  function notify() {
    if (!('Notification' in window) || Notification.permission !== 'granted' || !storageGet(noticeKey, false)) return;
    const seen = new Set(storageGet(notifiedKey, []));
    const fresh = data.videos.filter((video) => video.is_new && video.unread && !seen.has(video.id));
    if (!fresh.length) return;
    fresh.forEach((video) => seen.add(video.id));
    storageSet(notifiedKey, [...seen].slice(-1000));
    // Batch a backlog into one notification rather than flooding the desktop.
    const title = fresh.length === 1 ? '关注的 UP 主更新了' : `发现 ${fresh.length} 个视频更新`;
    const notification = new Notification(title, { body: fresh.slice(0, 3).map((v) => v.title).join('\n'), tag: 'mtd-creator-updates' });
    notification.onclick = () => { window.focus(); open(); notification.close(); };
  }
  function renderNotificationButton() {
    const enabled = 'Notification' in window && Notification.permission === 'granted' && storageGet(noticeKey, false);
    $('enableSubscriptionNotifications').textContent = enabled ? '关闭桌面提醒' : '启用桌面提醒';
  }
  function render() {
    const count = Number(data.unread_count || 0);
    $('subscriptionBadge').textContent = count > 99 ? '99+' : String(count);
    $('subscriptionBadge').classList.toggle('is-hidden', count === 0);
    $('subscriptionBadge').setAttribute('aria-label', `${count} 条未读更新`);
    $('subscriptionInboxCount').textContent = count > 99 ? '99+' : String(count);
    $('subscriptionInboxCount').classList.toggle('is-hidden', count === 0);
    $('subscriptionCount').textContent = `${data.subscriptions.length} 个`;
    $('subscriptionsServiceStatus').textContent = data.running
      ? (data.checking ? '正在检查更新…' : '后台检查已开启 · 服务关闭后暂停')
      : '后台检查未运行；可手动检查';
    if (selectedId && !data.subscriptions.some((s) => s.id === selectedId)) selectedId = null;
    renderChannels();
    renderVideos();
    renderNotificationButton();
  }
  function renderChannels() {
    const previousManagement = $('subscriptionChannelDetails').querySelector('.channel-management');
    const keepManagementOpen = previousManagement?.open && previousManagement.dataset.subscriptionId === selectedId;
    const query = $('subscriptionSearch').value.trim().toLocaleLowerCase();
    const subscriptions = data.subscriptions.filter((sub) => `${displayName(sub)} ${platformName(sub.platform)}`.toLocaleLowerCase().includes(query));
    $('allSubscriptionUpdates').classList.toggle('active', selectedId === null);
    $('allSubscriptionUpdates').setAttribute('aria-pressed', String(selectedId === null));
    $('subscriptionList').innerHTML = subscriptions.length ? subscriptions.map((sub) => {
      const unread = Number(sub.unread_count ?? unreadFor(sub.id));
      return `<button type="button" class="subscription-card ${sub.id === selectedId ? 'active' : ''}" data-subscription-id="${escapeHtml(sub.id)}" data-sub-action="select" aria-pressed="${sub.id === selectedId}">
        <span class="subscription-avatar ${escapeHtml(sub.platform)}" aria-hidden="true">${escapeHtml(displayName(sub).slice(0, 1))}</span>
        <span class="subscription-channel-name"><strong>${escapeHtml(displayName(sub))}</strong><small>${platformName(sub.platform)}${!sub.enabled ? ' · 已暂停' : sub.last_error ? ' · 检查失败' : ' · 已关注'}</small></span>
        ${unread ? `<span class="subscription-unread-count" aria-label="${unread} 条未读更新">${unread > 99 ? '99+' : unread}</span>` : ''}
      </button>`;
    }).join('') : `<div class="subscription-empty">${data.subscriptions.length ? '没有找到这个频道' : '还没有关注的频道<br>点击上方「添加关注」开始'}</div>`;
    const sub = data.subscriptions.find((s) => s.id === selectedId);
    $('subscriptionFeedTitle').textContent = sub ? displayName(sub) : '所有频道的更新';
    const details = $('subscriptionChannelDetails');
    details.classList.toggle('is-hidden', !sub);
    details.innerHTML = sub ? `<div class="channel-overview"><span class="subscription-platform ${escapeHtml(sub.platform)}">${platformName(sub.platform)}</span><span class="meta">${sub.enabled ? `每 ${sub.interval_minutes} 分钟检查 · ${sub.mode === 'auto' ? '自动下载' : '提醒后确认下载'}` : '已暂停检查'}${sub.transcribe ? ' · 自动转录' : ''}</span><a href="${escapeHtml(sub.url)}" target="_blank" rel="noreferrer" class="meta">频道主页 ↗</a></div>
      <details class="channel-management" data-subscription-id="${escapeHtml(sub.id)}"><summary>频道设置与检查</summary><div class="channel-management-body"><div class="meta">${sub.initialized ? `上次检查：${date(sub.last_checked)}` : '正在建立首次记录，历史视频不会自动下载'}</div>${sub.last_error ? `<div class="error">${escapeHtml(sub.last_error)}</div>` : ''}<div class="actions"><button type="button" class="small ghost" data-sub-action="check" ${!sub.enabled || sub.checking ? 'disabled' : ''}>${sub.checking ? '检查中…' : '检查'}</button><button class="small ghost" type="button" data-sub-action="toggle">${sub.enabled ? '暂停' : '恢复'}</button><button class="small ghost" type="button" data-sub-action="edit">编辑</button><button class="small ghost" type="button" data-sub-action="delete">取消关注</button></div></div></details>` : '';
    if (keepManagementOpen && sub) details.querySelector('.channel-management').open = true;
  }
  function renderVideos() {
    const filter = $('subscriptionVideoFilter').value;
    const scoped = data.videos.filter((v) => !selectedId || v.subscription_id === selectedId);
    const recent = filter === 'all'
      ? scoped.sort((a, b) => Number(b.published_at || 0) - Number(a.published_at || 0)
        || Number(b.first_seen || 0) - Number(a.first_seen || 0))
        .filter((video) => {
          const count = scoped.slice(0, scoped.indexOf(video) + 1)
            .filter((candidate) => candidate.subscription_id === video.subscription_id).length;
          return count <= 5;
        })
      : scoped;
    const videos = recent.filter((v) => filter === 'all' || (filter === 'unread' ? v.unread : filter === 'pending' ? v.state === 'pending' : v.is_new && v.state !== 'ignored'));
    const sub = data.subscriptions.find((s) => s.id === selectedId);
    const unread = sub ? Number(sub.unread_count ?? unreadFor(sub.id)) : data.unread_count || 0;
    $('subscriptionVideoCount').textContent = `${videos.length} 个视频 · ${unread} 条未读`;
    $('subscriptionVideos').innerHTML = videos.length ? videos.map((video) => {
      const sub = data.subscriptions.find((s) => s.id === video.subscription_id);
      const status = video.job_status;
      const retry = ['failed', 'cancelled'].includes(status);
      const jobExists = video.job_id && status;
      const label = jobExists ? window.statusLabel(status) : video.state === 'ignored' ? '已忽略' : video.is_new ? '新更新' : '历史视频';
      const thumb = video.thumbnail && /^https:\/\//.test(video.thumbnail)
        ? `<img class="subscription-thumbnail" src="${escapeHtml(video.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer" />`
        : '<div class="subscription-thumbnail subscription-thumbnail-placeholder" aria-hidden="true">▶</div>';
      return `<article class="subscription-video ${video.unread ? 'is-unread' : ''}" data-video-id="${escapeHtml(video.id)}">
        ${thumb}<div class="subscription-video-content"><a class="subscription-video-title" href="${escapeHtml(video.url)}" target="_blank" rel="noreferrer">${video.unread ? '<span class="unread-dot" aria-label="未读"></span>' : ''}${escapeHtml(video.title)}</a>
        <div class="meta">${escapeHtml(sub ? displayName(sub) : platformName(video.platform))} · ${video.duration ? `${escapeHtml(video.duration)} · ` : ''}${video.published_at ? date(video.published_at) : '发布时间未知'}</div>
        ${video.error || video.job_error ? `<div class="error">${escapeHtml(video.error || video.job_error)}</div>` : ''}
        <div class="subscription-video-foot"><span class="pill ${retry ? 'bad' : ''}">${escapeHtml(label)}</span><div class="actions">
        ${jobExists ? `<button class="small ghost" type="button" data-video-action="job">查看任务</button>` : ''}
        ${!jobExists || retry ? `<button class="small primary" type="button" data-video-action="download">${retry ? '重试下载' : '确认下载'}</button>` : ''}
        ${video.state === 'pending' ? '<button class="small ghost" type="button" data-video-action="dismiss">忽略</button>' : ''}
        </div></div></div></article>`;
    }).join('') : `<div class="subscription-feed-empty"><span aria-hidden="true">✓</span><h4>${!data.subscriptions.length ? '关注你喜欢的创作者' : filter === 'unread' ? '当前没有未读更新' : filter === 'pending' ? '当前没有待下载的视频' : filter === 'all' ? '还没有近期视频' : '暂时没有新投稿'}</h4><p>${!data.subscriptions.length ? '添加 YouTube 频道或 B 站 UP 主，新投稿会汇集在这里。' : filter === 'new' ? '有新投稿时，频道旁会亮起红色提醒。首次关注的历史投稿可在「全部近期视频」查看。' : '可以切换频道或筛选条件，查看其他视频。'}</p></div>`;
  }
  async function load() {
    if (loading) return;
    loading = true;
    try {
      const result = await request('api/subscriptions');
      data = { ...result, subscriptions: result.subscriptions || [], videos: result.videos || [] };
      render();
      notify();
    } catch (error) {
      $('subscriptionsServiceStatus').textContent = '无法连接后台检查服务';
      if (!view.classList.contains('is-hidden')) message(error.message, true);
    } finally { loading = false; }
  }
  function open() {
    if (!view.classList.contains('is-hidden')) return load();
    if (!window.confirmLeaveUnsavedChanges()) return;
    window.showImportView({ clearDraft: true });
    window.closeSettings();
    window.closeTranslate();
    window.closeClips();
    $('proofreadModal').classList.add('is-hidden');
    $('alignmentModal').classList.add('is-hidden');
    window.stopAlignmentPolling();
    window.stopPreviewPlayback();
    window.stopSubtitleSyncPolling();
    window.setVisible(view);
    load();
  }
  function close() {
    window.setVisible($('importView'));
    window.closeAiSettings();
  }
  function resetForm() {
    editingId = null;
    form.reset();
    $('subscriptionFormMessage').textContent = '';
    $('subscriptionSource').readOnly = false;
    $('subscriptionPlatform').disabled = false;
    $('subscriptionFormTitle').textContent = '添加关注';
    $('subscriptionSubmit').textContent = '添加关注';
    $('subscriptionCancelEdit').textContent = '取消';
    updatePlaceholder();
  }
  function updatePlaceholder() {
    $('subscriptionSource').placeholder = $('subscriptionPlatform').value === 'youtube' ? '频道链接、@账号或 UC 频道 ID' : 'UP 主主页链接或数字 UID';
  }
  function edit(sub) {
    editingId = sub.id;
    $('subscriptionPlatform').value = sub.platform;
    $('subscriptionPlatform').disabled = true;
    $('subscriptionSource').value = sub.url;
    $('subscriptionSource').readOnly = true;
    $('subscriptionName').value = sub.name;
    $('subscriptionMode').value = sub.mode;
    $('subscriptionInterval').value = sub.interval_minutes;
    $('subscriptionBrowser').value = sub.cookies_browser;
    $('subscriptionTranscribe').checked = sub.transcribe;
    $('subscriptionFormTitle').textContent = '编辑关注';
    $('subscriptionSubmit').textContent = '保存设置';
    $('subscriptionCancelEdit').classList.remove('is-hidden');
    $('subscriptionDialog').showModal();
  }
  async function act(button, callback) {
    if (busy) return;
    busy = true;
    button.disabled = true;
    try { await callback(); await load(); }
    catch (error) { message(error.message, true); }
    finally { busy = false; button.disabled = false; }
  }
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    act($('subscriptionSubmit'), async () => {
      message(editingId ? '保存设置中…' : '正在添加并读取频道近期视频…');
      const payload = { name: $('subscriptionName').value, mode: $('subscriptionMode').value,
        interval_minutes: Number($('subscriptionInterval').value), transcribe: $('subscriptionTranscribe').checked,
        cookies_browser: $('subscriptionBrowser').value };
      let result;
      if (editingId) result = await request(`api/subscriptions/${editingId}`, 'PUT', payload);
      else result = await request('api/subscriptions', 'POST', { ...payload, platform: $('subscriptionPlatform').value, source: $('subscriptionSource').value });
      resetForm();
      $('subscriptionDialog').close();
      message(result.last_error ? `已保存关注，首次检查未成功：${result.last_error}` : '关注已保存。首次记录的历史视频不会自动下载。', !!result.last_error);
    });
  });
  function handleChannelAction(event) {
    const button = event.target.closest('button[data-sub-action]');
    if (!button) return;
    const sub = data.subscriptions.find((s) => s.id === button.closest('[data-subscription-id]').dataset.subscriptionId);
    if (!sub) return;
    const action = button.dataset.subAction;
    if (action === 'select') {
      selectedId = sub.id;
      renderChannels();
      renderVideos();
      return;
    }
    if (action === 'edit') return edit(sub);
    if (action === 'delete' && !window.confirm(`取消关注「${displayName(sub)}」？已下载的视频和任务会保留。`)) return;
    act(button, async () => {
      if (action === 'check') {
        message(`正在检查 ${displayName(sub)}…`);
        const result = await request(`api/subscriptions/${sub.id}/check`, 'POST');
        message(result.error || (result.scheduled ? '已安排检查，请稍后查看结果。' : `检查完成，发现 ${result.new_count || 0} 个新视频。`), !!result.error);
      } else if (action === 'toggle') {
        await request(`api/subscriptions/${sub.id}`, 'PUT', { enabled: !sub.enabled });
        message(sub.enabled ? '已暂停自动检查。' : '已恢复自动检查。');
      } else if (action === 'delete') {
        await request(`api/subscriptions/${sub.id}`, 'DELETE');
        if (editingId === sub.id) resetForm();
        message('已取消关注。');
      }
    });
  }
  $('subscriptionList').addEventListener('click', handleChannelAction);
  $('subscriptionChannelDetails').addEventListener('click', handleChannelAction);
  $('subscriptionVideos').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-video-action]');
    if (!button) return;
    const video = data.videos.find((v) => v.id === button.closest('[data-video-id]').dataset.videoId);
    if (!video) return;
    act(button, async () => {
      if (button.dataset.videoAction === 'job') {
        window.setVisible($('importView'));
        await window.refreshJobs({ keepSelection: true });
        await window.selectJob(video.job_id);
      } else if (button.dataset.videoAction === 'download') {
        const job = await request(`api/subscription-videos/${encodeURIComponent(video.id)}/download`, 'POST');
        message(`已加入${job.download_only ? '下载' : '下载和转录'}队列。`);
        await window.refreshJobs({ keepSelection: true, background: true });
      } else {
        await request(`api/subscription-videos/${encodeURIComponent(video.id)}/dismiss`, 'POST');
        message('已忽略这条更新。');
      }
    });
  });
  $('checkAllSubscriptions').addEventListener('click', () => act($('checkAllSubscriptions'), async () => {
    message('正在检查已启用的频道…');
    const result = await request('api/subscriptions/check', 'POST');
    const results = result.results || [];
    const errors = results.filter((r) => r.error).length;
    message(`检查完成，发现 ${results.reduce((n, r) => n + (r.new_count || 0), 0)} 个新视频${errors ? `，${errors} 个频道检查失败，请查看频道详情` : ''}。`, errors > 0);
  }));
  $('readSubscriptionUpdates').addEventListener('click', () => act($('readSubscriptionUpdates'), async () => { await request('api/subscriptions/read', 'POST'); message('已标记全部更新为已读。'); }));
  $('enableSubscriptionNotifications').addEventListener('click', async () => {
    if (!('Notification' in window)) return message('浏览器不支持桌面通知，可使用顶部更新提示。', true);
    if (storageGet(noticeKey, false)) storageSet(noticeKey, false);
    else {
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') return message('未开启通知权限，顶部更新提示仍可使用。', true);
      storageSet(noticeKey, true);
    }
    renderNotificationButton();
    notify();
  });
  $('openSubscriptions').addEventListener('click', open);
  $('backFromSubscriptions').addEventListener('click', close);
  $('addSubscription').addEventListener('click', () => { resetForm(); $('subscriptionDialog').showModal(); });
  $('closeSubscriptionDialog').addEventListener('click', () => $('subscriptionDialog').close());
  $('subscriptionCancelEdit').addEventListener('click', () => $('subscriptionDialog').close());
  $('subscriptionDialog').addEventListener('close', () => { if (!busy) resetForm(); });
  $('subscriptionSearch').addEventListener('input', renderChannels);
  $('allSubscriptionUpdates').addEventListener('click', () => { selectedId = null; renderChannels(); renderVideos(); });
  $('subscriptionPlatform').addEventListener('change', updatePlaceholder);
  $('subscriptionVideoFilter').addEventListener('change', renderVideos);
  $('subscriptionVideos').addEventListener('error', (event) => {
    if (event.target.matches('img.subscription-thumbnail')) {
      const placeholder = document.createElement('div');
      placeholder.className = 'subscription-thumbnail subscription-thumbnail-placeholder';
      placeholder.textContent = '▶';
      placeholder.setAttribute('aria-hidden', 'true');
      event.target.replaceWith(placeholder);
    }
  }, true);
  window.MtdSubscriptions = { open, refresh: load };
  load();
  setInterval(load, 15000);
})();
