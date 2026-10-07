/* Application shell. The editor keeps its existing, full-width workspace. */
(() => {
  const get = (id) => document.getElementById(id);
  const importPage = get('importView');
  const pageNames = { home: '首页', tasks: '任务中心', create: '新建任务' };

  window.shellJobs = (items) => {
    get('hubTotal').textContent = items.length;
    get('navJobCount').textContent = items.length;
    get('hubRunning').textContent = items.filter((job) => RUNNING_STATES.has(job.status)).length;
    get('hubEditable').textContent = items.filter((job) => EDIT_STATES.has(job.status)).length;
    if (importPage.dataset.shellMode !== 'tasks') return items.slice(0, 6);
    const query = get('hubTaskSearch').value.trim().toLocaleLowerCase();
    const filter = get('hubTaskFilter').value;
    return items.filter((job) => {
      const matchesText = `${job.media_name || ''} ${job.id}`.toLocaleLowerCase().includes(query);
      const matchesState = filter === 'all'
        || (filter === 'running' && RUNNING_STATES.has(job.status))
        || (filter === 'editable' && EDIT_STATES.has(job.status))
        || (filter === 'failed' && ['failed', 'error', 'cancelled', 'canceled'].includes(job.status));
      return matchesText && matchesState;
    });
  };

  window.updateShell = (view) => {
    const editor = view.id === 'workbench';
    document.body.classList.toggle('editor-workspace', editor);
    const mode = importPage.dataset.shellMode;
    const page = view === importPage ? mode : view.id;
    get('shellPageTitle').textContent = pageNames[page] || {
      subscriptionsView: '关注更新', aiSettingsView: 'AI 服务',
      processingView: '任务进度', downloadedView: '下载完成', workbench: '字幕编辑',
    }[page] || '工作空间';
    document.querySelectorAll('.shell-nav').forEach((button) => {
      const active = button.dataset.shellPage === page
        || (button.id === 'openSubscriptions' && page === 'subscriptionsView')
        || (button.id === 'openAiSettings' && page === 'aiSettingsView');
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    get('hubTitle').textContent = mode === 'tasks' ? '所有工作，都在这里。'
      : mode === 'create' ? '开始一项新的转写。' : '从这里，继续你的工作。';
    get('hubDescription').textContent = mode === 'tasks' ? '查找素材、查看进度，或进入项目继续编辑。'
      : mode === 'create' ? '选择文件或视频链接，设置好参数后开始。' : '导入新素材，或接着完成上一次的字幕与剪辑。';
    get('hubJobsTitle').textContent = mode === 'tasks' ? '全部任务' : '最近任务';
  };

  function navigate(mode, tab) {
    if (!confirmLeaveUnsavedChanges()) return;
    showImportView({ clearDraft: false, shellMode: mode });
    if (tab) document.querySelector(`.tab-btn[data-tab="${tab}"]`).click();
    importPage.scrollTop = 0;
  }

  document.querySelectorAll('[data-shell-page]').forEach((button) => {
    button.addEventListener('click', () => navigate(button.dataset.shellPage));
  });
  document.querySelectorAll('[data-shell-create]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!confirmLeaveUnsavedChanges()) return;
      showImportView({ clearDraft: true, shellMode: 'create' });
      document.querySelector(`.tab-btn[data-tab="${button.dataset.shellCreate}"]`).click();
      importPage.scrollTop = 0;
    });
  });
  ['openAiSettings'].forEach((id) => {
    get(id).addEventListener('click', (event) => {
      if (!confirmLeaveUnsavedChanges()) event.stopImmediatePropagation();
    }, true);
  });
  get('editorSubscriptions').addEventListener('click', () => get('openSubscriptions').click());
  get('editorAiSettings').addEventListener('click', () => get('openAiSettings').click());
  get('hubTaskSearch').addEventListener('input', renderJobList);
  get('hubTaskFilter').addEventListener('change', renderJobList);
  window.updateShell(document.querySelector('section.content > .view:not(.is-hidden)'));
  renderJobList();
})();
