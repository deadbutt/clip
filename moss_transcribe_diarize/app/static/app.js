const RUNNING_STATES = new Set(['queued', 'downloading', 'loading_model', 'transcribing', 'postprocessing', 'labeling_speakers', 'translating', 'proofreading', 'rendering']);
const EDIT_STATES = new Set(['waiting_review', 'rendering', 'done']);
const TERMINAL_STATES = new Set(['waiting_review', 'done', 'failed', 'cancelled']);
const fileInput = document.querySelector('#file');
const importTitleEl = document.querySelector('#importTitle');
const rerunSourceEl = document.querySelector('#rerunSource');
const promptInput = document.querySelector('#prompt');
const hotwordsInput = document.querySelector('#hotwordsInput');
const uploadHotwordsInput = document.querySelector('#uploadHotwords');
const speakerCountInput = document.querySelector('#speakerCount');
const diarizationBackendSelect = document.querySelector('#diarizationBackend');
const forceTranscribeInput = document.querySelector('#forceTranscribe');
const rerunParamsEl = document.querySelector('#rerunParams');
const rerunPromptInput = document.querySelector('#rerunPrompt');
const rerunSpeakerCountInput = document.querySelector('#rerunSpeakerCount');
const rerunDiarizationSelect = document.querySelector('#rerunDiarization');
let storedDefaults = { prompt: '', speakerCount: '', backend: 'none' };
const urlInput = document.querySelector('#urlInput');
const pasteUrlBtn = document.querySelector('#pasteUrlBtn');
const cookiesBrowserSelect = document.querySelector('#cookiesBrowser');
const cookiesFileInput = document.querySelector('#cookiesFile');
const checkCookiesBtn = document.querySelector('#checkCookiesBtn');
const cookiesResultEl = document.querySelector('#cookiesResult');
const uploadBtn = document.querySelector('#upload');
const newTaskBtn = document.querySelector('#newTask');
const refreshJobsBtn = document.querySelector('#refreshJobs');
const deleteCurrentBtn = document.querySelector('#deleteCurrent');
const openNewBtn = document.querySelector('#openNew');
const backFromProcessingBtn = document.querySelector('#backFromProcessing');
const backToTasksBtn = document.querySelector('#backToTasks');
const openSettingsBtn = document.querySelector('#openSettings');
const closeSettingsBtn = document.querySelector('#closeSettings');
const settingsModal = document.querySelector('#settingsModal');
const openTranslateBtn = document.querySelector('#openTranslate');
const closeTranslateBtn = document.querySelector('#closeTranslate');
const translateModal = document.querySelector('#translateModal');
const openClipsBtn = document.querySelector('#openClips');
const closeClipsBtn = document.querySelector('#closeClips');
const clipsModal = document.querySelector('#clipsModal');
const pendingListEl = document.querySelector('#pendingList');
const saveBtn = document.querySelector('#save');
const syncSubtitlesBtn = document.querySelector('#syncSubtitles');
const undoBtn = document.querySelector('#undoBtn');
const redoBtn = document.querySelector('#redoBtn');
const renderBtn = document.querySelector('#render');
const rerunBtn = document.querySelector('#rerun');
const addSegmentBtn = document.querySelector('#addSegment');
const deleteSegmentBtn = document.querySelector('#deleteSegment');
const shiftSegmentLeftBtn = document.querySelector('#shiftSegmentLeft');
const shiftSegmentRightBtn = document.querySelector('#shiftSegmentRight');
const nudgeStartLeftBtn = document.querySelector('#nudgeStartLeft');
const nudgeStartRightBtn = document.querySelector('#nudgeStartRight');
const nudgeEndLeftBtn = document.querySelector('#nudgeEndLeft');
const nudgeEndRightBtn = document.querySelector('#nudgeEndRight');
const settingsSaveBtn = document.querySelector('#settingsSave');
const saveStatusEl = document.querySelector('#saveStatus');
const importView = document.querySelector('#importView');
const processingView = document.querySelector('#processingView');
const workbench = document.querySelector('#workbench');
const runtimeEl = document.querySelector('#runtime');
const jobListEl = document.querySelector('#jobList');
const jobCountEl = document.querySelector('#jobCount');
const importErrorEl = document.querySelector('#importError');
const processTitleEl = document.querySelector('#processTitle');
const processNameEl = document.querySelector('#processName');
const processMetaEl = document.querySelector('#processMeta');
const processBarEl = document.querySelector('#processBar');
const processErrorEl = document.querySelector('#processError');
const selectedNameEl = document.querySelector('#selectedName');
const taskStatusEl = document.querySelector('#taskStatus');
const taskUsageEl = document.querySelector('#taskUsage');
const taskParamsEl = document.querySelector('#taskParams');
const taskNoticeEl = document.querySelector('#taskNotice');
const renderProgressMetaEl = document.querySelector('#renderProgressMeta');
const renderProgressTextEl = document.querySelector('#renderProgressText');
const renderProgressEl = document.querySelector('#renderProgress');
const renderProgressBarEl = document.querySelector('#renderProgressBar');
const modelInfoEl = document.querySelector('#modelinfo');
const tbody = document.querySelector('#segments');
const tableWrap = document.querySelector('.table-column .table-wrap');
const cancelCurrentBtn = document.querySelector('#cancelCurrent');
const searchQueryInput = document.querySelector('#searchQuery');
const searchModeSelect = document.querySelector('#searchMode');
const searchCountEl = document.querySelector('#searchCount');
const searchPrevBtn = document.querySelector('#searchPrev');
const searchNextBtn = document.querySelector('#searchNext');
const replaceTextInput = document.querySelector('#replaceText');
const replaceAllBtn = document.querySelector('#replaceAll');
// 用户手动滚动字幕表格后的一段时间内，禁止视频播放进度自动跟随滚动
let tableUserScrollUntil = 0;
const TABLE_USER_SCROLL_HOLD_MS = 3000;
const markTableUserScroll = () => { tableUserScrollUntil = performance.now() + TABLE_USER_SCROLL_HOLD_MS; };
tableWrap.addEventListener('wheel', markTableUserScroll, { passive: true });
tableWrap.addEventListener('pointerdown', markTableUserScroll);
tableWrap.addEventListener('touchstart', markTableUserScroll, { passive: true });
const speakerMapEl = document.querySelector('#speakerMap');
const videoStage = document.querySelector('#videoStage');
const videoShell = document.querySelector('.video-shell');
const preview = document.querySelector('#preview');
const maskPreviewVideo = document.querySelector('#maskPreviewVideo');
const sourceMaskOverlay = document.querySelector('#sourceMaskOverlay');
const subtitleOverlay = document.querySelector('#subtitleOverlay');
const timelineScroll = document.querySelector('#timelineScroll');
const timelineTrack = document.querySelector('#timelineTrack');
const timelineRuler = document.querySelector('#timelineRuler');
const timelineLane = document.querySelector('#timelineLane');
const timelinePlayhead = document.querySelector('#timelinePlayhead');
const timelineGuide = document.querySelector('#timelineGuide');
const timelineClipRange = document.querySelector('#timelineClipRange');
const timelineMeta = document.querySelector('#timelineMeta');
const translateModeSelect = document.querySelector('#translateMode');
const targetLanguageInput = document.querySelector('#targetLanguage');
const translateZhBtn = document.querySelector('#translateZh');
const translateStatusEl = document.querySelector('#translateStatus');
const translateModelStatusEl = document.querySelector('#translateModelStatus');
const translateProtectedTermsInput = document.querySelector('#translateProtectedTerms');
const translateProgressMetaEl = document.querySelector('#translateProgressMeta');
const translateProgressTextEl = document.querySelector('#translateProgressText');
const translateProgressEl = document.querySelector('#translateProgress');
const translateProgressBarEl = document.querySelector('#translateProgressBar');
const restoreTranslationBtn = document.querySelector('#restoreTranslation');
const translationReviewEl = document.querySelector('#translationReview');
const translationReviewMetaEl = document.querySelector('#translationReviewMeta');
const translationReviewListEl = document.querySelector('#translationReviewList');
const clipTitleInput = document.querySelector('#clipTitleInput');
const clipStartInput = document.querySelector('#clipStart');
const clipEndInput = document.querySelector('#clipEnd');
const clipDurationInput = document.querySelector('#clipDuration');
const clipMinDurationInput = document.querySelector('#clipMinDuration');
const clipTargetDurationInput = document.querySelector('#clipTargetDuration');
const clipMaxDurationInput = document.querySelector('#clipMaxDuration');
const useActiveSegmentBtn = document.querySelector('#useActiveSegment');
const findClipsBtn = document.querySelector('#findClips');
const findClipsRulesBtn = document.querySelector('#findClipsRules');
const renderClipBtn = document.querySelector('#renderClip');
const renderClipQueueBtn = document.querySelector('#renderClipQueue');
const clipMoveBackBtn = document.querySelector('#clipMoveBack');
const clipMoveForwardBtn = document.querySelector('#clipMoveForward');
const clipStartEarlierBtn = document.querySelector('#clipStartEarlier');
const clipEndLaterBtn = document.querySelector('#clipEndLater');
const clipStatusEl = document.querySelector('#clipStatus');
const clipListEl = document.querySelector('#clipList');
const clipQueueListEl = document.querySelector('#clipQueueList');
const clipCandidateCountEl = document.querySelector('#clipCandidateCount');
const clipQueueCountEl = document.querySelector('#clipQueueCount');
const clipModelStatusEl = document.querySelector('#clipModelStatus');
const openProofreadBtn = document.querySelector('#openProofread');
const closeProofreadBtn = document.querySelector('#closeProofread');
const proofreadModal = document.querySelector('#proofreadModal');
const proofreadModelStatusEl = document.querySelector('#proofreadModelStatus');
const proofreadRunBtn = document.querySelector('#proofreadRun');
const proofreadApplyBtn = document.querySelector('#proofreadApply');
const proofreadStatusEl = document.querySelector('#proofreadStatus');
const proofreadSelectionMetaEl = document.querySelector('#proofreadSelectionMeta');
const proofreadProgressMetaEl = document.querySelector('#proofreadProgressMeta');
const proofreadProgressTextEl = document.querySelector('#proofreadProgressText');
const proofreadProgressEl = document.querySelector('#proofreadProgress');
const proofreadProgressBarEl = document.querySelector('#proofreadProgressBar');
const proofreadTermSectionEl = document.querySelector('#proofreadTermSection');
const proofreadTermsAllEl = document.querySelector('#proofreadTermsAll');
const proofreadTermsMetaEl = document.querySelector('#proofreadTermsMeta');
const proofreadTermsListEl = document.querySelector('#proofreadTermsList');
const proofreadTypoSectionEl = document.querySelector('#proofreadTypoSection');
const proofreadTyposAllEl = document.querySelector('#proofreadTyposAll');
const proofreadTyposMetaEl = document.querySelector('#proofreadTyposMeta');
const proofreadTyposListEl = document.querySelector('#proofreadTyposList');
const proofreadAlignmentSectionEl = document.querySelector('#proofreadAlignmentSection');
const proofreadAlignmentListEl = document.querySelector('#proofreadAlignmentList');
const proofreadAlignmentMetaEl = document.querySelector('#proofreadAlignmentMeta');
const proofreadAlignmentAllEl = document.querySelector('#proofreadAlignmentAll');
const llmProfileListEl = document.querySelector('#llmProfileList');
const llmProfileAddBtn = document.querySelector('#llmProfileAdd');
const llmProfileEditorEl = document.querySelector('#llmProfileEditor');
const llmProfileNameInput = document.querySelector('#llmProfileName');
const llmProfileProviderSelect = document.querySelector('#llmProfileProvider');
const llmProfileBaseUrlInput = document.querySelector('#llmProfileBaseUrl');
const llmProfileModelInput = document.querySelector('#llmProfileModel');
const llmProfileApiKeyInput = document.querySelector('#llmProfileApiKey');
const llmProfileDisableThinkingSelect = document.querySelector('#llmProfileDisableThinking');
const llmProfileSaveBtn = document.querySelector('#llmProfileSave');
const llmProfileCancelBtn = document.querySelector('#llmProfileCancel');
const llmProfileTestBtn = document.querySelector('#llmProfileTest');
const llmProfileTestResultEl = document.querySelector('#llmProfileTestResult');
let llmProfiles = [];
let llmEditingProfileId = null;
let proofreadResult = null;
let proofreadPollTimer = null;
const hotwordsGlossaryViewEl = document.querySelector('#hotwordsGlossaryView');
const hotwordsGlossaryEditBtn = document.querySelector('#hotwordsGlossaryEdit');
const hotwordsGlossaryEditorEl = document.querySelector('#hotwordsGlossaryEditor');
const hotwordsGlossaryTextEl = document.querySelector('#hotwordsGlossaryText');
const hotwordsGlossarySaveBtn = document.querySelector('#hotwordsGlossarySave');
const hotwordsGlossaryCancelBtn = document.querySelector('#hotwordsGlossaryCancel');
let hotwordsGlossary = [];
let jobs = [];
let currentJob = null;
let rerunDraftJob = null;
let pendingUploads = [];
let pendingIdCounter = 0;
let pollTimer = null;
let runtimeChecked = false;
let ffmpegAvailable = false;
let translatorAvailable = false;
let translatorInfo = {};
let activeSegmentIndex = -1;
let assPlayRes = { x: 1920, y: 1080 };
let layoutFitFrame = 0;
let editorDirty = false;
let saveStatusTimer = 0;
let speakerNameMap = {};
// 说话人 → 自定义字幕颜色(html #rrggbb)。未指定的说话人用调色板默认色。
let speakerColorOverrides = {};
let timelineDragging = false;
let currentPixelsPerSecond = 12;
let segmentDragState = null;
let clipDragState = null;
let subtitleSyncTimer = 0;
let subtitleSyncInFlight = false;
// segments 接口的 ETag({jobId, etag}):文件没变时服务器回 304,跳过整段解析/深比较/重渲染。
// 切换任务或本地写操作(保存/拆分/合并/替换等走 renderSegments)时置空,下次轮询全量拉一次再续上。
let segmentsEtag = null;
let cachedSegments = null;
let undoStack = [];
let redoStack = [];
// 撤销栈所属的任务 id：切换任务时据此清空，防止 A 任务的历史快照被撤销进 B 任务
let undoJobId = null;
const MAX_UNDO = 50;
let cachedTimelineSegments = [];
let cachedTimelineLayout = { lanes: new Map(), count: 1 };
let syncActiveFrame = 0;
let lastSyncedTime = -1;
let tableRenderFrame = 0;
let timelineRenderFrame = 0;
let timelineAutoScrollFrame = 0;
let timelineAutoScrollPointer = null;
let timelineAutoScrollMode = '';
let timelineFollowHoldUntil = 0;
let dismissedTranslationReviewItems = new Set();
let translationReviewJobId = '';
let dismissedAlignmentItems = new Set();
let clipQueueJobId = '';
let selectedClips = [];
let activeClipId = '';
const SEGMENT_EDGE_PX = 8;
const SEGMENT_DRAG_THRESHOLD = 3;
const SEGMENT_DRAG_SENSITIVITY = 5;
const SNAP_PX = 18;
const TABLE_ROW_HEIGHT = 52;
const TABLE_BUFFER_ROWS = 24;
const TIMELINE_BUFFER_PX = 700;
const assFontLineHeightFactor = 1.448;
const speakerPalette = ['#ffffff', '#ffe75b', '#8ff286', '#ffa7bb', '#ffd700', '#6bb5ff', '#db8eff', '#d8d8d8'];
const RENDER_PROGRESS_BASE = 0.95;
const RENDER_PROGRESS_SPAN = 0.049;

function apiUrl(path) {
  const clean = String(path).replace(/^[/]+/, '');
  const basePath = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
  return new URL(clean, window.location.origin + basePath).toString();
}

function setPreviewSource(src) {
  preview.src = src;
  maskPreviewVideo.removeAttribute('src');
  maskPreviewVideo.load();
}

async function refreshRuntime() {
  try {
    const res = await fetch(apiUrl('api/runtime'), { cache: 'no-store' });
    if (!res.ok) throw new Error('runtime status ' + res.status);
    const data = await res.json();
    runtimeChecked = true;
    ffmpegAvailable = !!(data.ffmpeg && data.ffmpeg.available);
    translatorAvailable = !!(data.translator && data.translator.available);
    translatorInfo = data.translator || {};
    runtimeEl.textContent = ffmpegAvailable ? 'FFmpeg 可用' : 'FFmpeg 缺失';
    runtimeEl.className = 'pill ' + (ffmpegAvailable ? 'ok' : 'bad');
    updateTranslateAction();
    updateClipActions();
    updateRenderAction(currentJob);
    applyInferenceDefaults(data.inference || {});
    applySpeakerDefaults(data.speaker_labeling || {});
    applyTranslatorDefaults(data.translator || {});
    renderModelInfo(data.model || {});
  } catch (err) {
    runtimeChecked = true;
    ffmpegAvailable = false;
    translatorAvailable = false;
    translatorInfo = {};
    runtimeEl.textContent = 'API 连接失败';
    runtimeEl.className = 'pill bad';
    updateTranslateAction();
    updateClipActions();
    updateRenderAction(currentJob);
    importErrorEl.textContent = '无法连接 api/runtime，请确认页面来自 mtd-subtitle-web 服务。';
  }
}

function applyInferenceDefaults(defaults) {
 if (defaults.prompt) {
   storedDefaults.prompt = defaults.prompt;
   if (!promptInput.value) promptInput.value = defaults.prompt;
 }
}

function applySpeakerDefaults(defaults) {
 if (defaults.default_speaker_count) {
   storedDefaults.speakerCount = String(defaults.default_speaker_count);
   if (!speakerCountInput.value) speakerCountInput.value = defaults.default_speaker_count;
 }
 if (defaults.default_backend) {
   storedDefaults.backend = defaults.default_backend;
   diarizationBackendSelect.value = defaults.default_backend;
 }
}

function applyTranslatorDefaults(defaults) {
  if (!translateProtectedTermsInput || translateProtectedTermsInput.value) return;
  const terms = Array.isArray(defaults.protected_terms) ? defaults.protected_terms : [];
  translateProtectedTermsInput.value = terms.join(', ');
}

function normalizedSpeakerCount(value) {
 if (value === '' || value == null) return '';
 const count = Number(value);
 if (!Number.isFinite(count) || count <= 0) return '';
 return String(Math.max(1, Math.min(10, Math.round(count))));
}

function renderModelInfo(model) {
  const parts = [];
  if (model.path) {
    const pathParts = String(model.path).split('/');
    parts.push(pathParts.slice(-2).join('/'));
  }
  if (model.device) parts.push(model.device);
  if (model.dtype) parts.push(model.dtype);
  const processor = model.processor || {};
  if (processor.time_marker_every_seconds) parts.push('time marker ' + processor.time_marker_every_seconds + 's');
  if (modelInfoEl) modelInfoEl.textContent = parts.join(' · ');
}

function scheduleLayoutFit() {
  if (layoutFitFrame) cancelAnimationFrame(layoutFitFrame);
  layoutFitFrame = requestAnimationFrame(() => {
    layoutFitFrame = 0;
    fitVideoStageToMedia();
  });
}

function openSettings() { settingsModal.classList.remove('is-hidden'); }
function closeSettings() { settingsModal.classList.add('is-hidden'); }
function openTranslate() {
  updateTranslateAction();
  translateModal.classList.remove('is-hidden');
}
function closeTranslate() { translateModal.classList.add('is-hidden'); }
function openClips() { ensureClipQueueForJob(); updateClipActions(); updateTimelineClipRange(); clipsModal.classList.remove('is-hidden'); }
function closeClips() { clipsModal.classList.add('is-hidden'); }

function startSubtitleSyncPolling() {
  stopSubtitleSyncPolling();
  if (!currentJob || !EDIT_STATES.has(currentJob.status)) return;
  subtitleSyncTimer = setInterval(() => {
    if (!currentJob || editorDirty || subtitleSyncInFlight || !EDIT_STATES.has(currentJob.status)) return;
    loadSegments(currentJob.id, { preserveSelection: true }).catch(() => {});
  }, 2000);
}

function stopSubtitleSyncPolling() {
  if (!subtitleSyncTimer) return;
  clearInterval(subtitleSyncTimer);
  subtitleSyncTimer = 0;
}

function setSaveState(state, message) {
  if (saveStatusTimer) {
    clearTimeout(saveStatusTimer);
    saveStatusTimer = 0;
  }
  saveStatusEl.className = 'save-status ' + state;
  saveStatusEl.textContent = message;
  const showButton = state === 'dirty' || state === 'saving' || state === 'error';
  saveBtn.classList.toggle('is-hidden', !showButton);
  saveBtn.classList.toggle('primary', showButton);
  saveBtn.classList.toggle('saved', false);
  saveBtn.disabled = state === 'saving' || !currentJob;
  if (state === 'dirty') saveBtn.textContent = '保存修改';
  else if (state === 'saving') saveBtn.textContent = '保存中...';
  else if (state === 'error') saveBtn.textContent = '重试保存';
  else saveBtn.textContent = '保存修改';
}

function setEditorDirty(dirty) {
  editorDirty = dirty;
  if (syncSubtitlesBtn) syncSubtitlesBtn.disabled = dirty || !currentJob;
  if (syncSubtitlesBtn) syncSubtitlesBtn.disabled = dirty || !currentJob;
  if (dirty) setSaveState('dirty', '有未保存修改');
  else setSaveState('saved', '已保存');
}

function markEditorDirty() {
  if (!currentJob) return;
  setEditorDirty(true);
}

function pushUndoSnapshot() {
  if (!cachedSegments) return;
  undoJobId = currentJob ? currentJob.id : undoJobId;
  const snapshot = JSON.stringify(cachedSegments);
  if (undoStack.length > 0 && undoStack[undoStack.length - 1] === snapshot) return;
  undoStack.push(snapshot);
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack = [];
  updateUndoRedoButtons();
}

function resetUndoHistory() {
  undoStack = [];
  redoStack = [];
  undoJobId = null;
  updateUndoRedoButtons();
}

function markEditorDirtyWithUndo() {
  // 每轮修改（从已保存状态出发的第一次变更）前推一次基线快照。
  // 不能只依赖各调用点自觉 pushUndoSnapshot：样式/说话人名这类不进快照的修改
  // 也会置脏 editorDirty，若它们先发生，后续文本编辑的基线就永远推不进去。
  if (!editorDirty) pushUndoSnapshot();
  markEditorDirty();
}

async function undoEdit() {
  if (undoStack.length === 0 || !cachedSegments) return;
  redoStack.push(JSON.stringify(cachedSegments));
  const prev = JSON.parse(undoStack.pop());
  renderSegments(prev);
  await saveSegments(true);
  updateUndoRedoButtons();
}

async function redoEdit() {
  if (redoStack.length === 0 || !cachedSegments) return;
  undoStack.push(JSON.stringify(cachedSegments));
  const next = JSON.parse(redoStack.pop());
  renderSegments(next);
  await saveSegments(true);
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  if (undoBtn) undoBtn.disabled = undoStack.length === 0;
  if (redoBtn) redoBtn.disabled = redoStack.length === 0;
}

document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.toggle('active', b.dataset.tab === target));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === target + 'Tab'));
    if (target === 'url' && rerunDraftJob) resetImportMode();
    updateUploadBtnLabel();
  });
});

pasteUrlBtn.addEventListener('click', async () => {
  try {
    const text = await navigator.clipboard.readText();
    urlInput.value = text.trim();
    urlInput.focus();
  } catch (err) {
    urlInput.focus();
  }
});

urlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitUrlDownload();
  }
});

let isSubmitting = false;
uploadBtn.addEventListener('click', async () => {
  if (isSubmitting) return;
  if (document.querySelector('#urlTab.active') && urlInput.value.trim()) {
    isSubmitting = true;
    await submitUrlDownload();
    isSubmitting = false;
    return;
  }
  if (rerunDraftJob) {
    isSubmitting = true;
    await startRerunDraft();
    isSubmitting = false;
    return;
  }
  if (!pendingUploads.length) return;
  isSubmitting = true;
  uploadBtn.disabled = true;
  importErrorEl.textContent = '';
  const total = pendingUploads.length;
  let created = 0;
  for (const item of pendingUploads) {
    uploadBtn.textContent = '上传中 ' + (created + 1) + '/' + total;
    const form = new FormData();
    form.append('file', item.file);
    form.append('prompt', item.prompt);
    if (item.hotwords) form.append('hotwords', item.hotwords);
    if (item.speakerCount) form.append('speaker_count', item.speakerCount);
    form.append('diarization_backend', item.diarizationBackend || 'auto');
    try {
      const res = await fetch(apiUrl('api/jobs'), { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) {
        importErrorEl.textContent = (data && data.detail) || '上传失败';
        break;
      }
      created += 1;
    } catch (err) {
      importErrorEl.textContent = '上传失败：' + (err && err.message ? err.message : String(err));
      break;
    }
  }
  pendingUploads = [];
  renderPendingList();
  uploadBtn.disabled = false;
  isSubmitting = false;
  updateUploadBtnLabel();
  await refreshJobs({ keepSelection: true });
});

async function submitUrlDownload() {
  const url = urlInput.value.trim();
  if (!url) {
    importErrorEl.textContent = '请输入视频链接';
    return;
  }
  importErrorEl.textContent = '';
  uploadBtn.disabled = true;
  uploadBtn.textContent = '提交中...';
  const form = new FormData();
  form.append('url', url);
  form.append('cookies_browser', cookiesBrowserSelect.value);
  if (promptInput.value) form.append('prompt', promptInput.value);
  if (hotwordsInput.value.trim()) form.append('hotwords', hotwordsInput.value.trim());
  if (speakerCountInput.value) form.append('speaker_count', speakerCountInput.value);
  form.append('diarization_backend', diarizationBackendSelect.value || 'auto');
  if (forceTranscribeInput && forceTranscribeInput.checked) form.append('force_transcribe', '1');
  if (cookiesFileInput.files && cookiesFileInput.files[0]) {
    form.append('cookies_file', cookiesFileInput.files[0]);
  }
  try {
    const res = await fetch(apiUrl('api/jobs/url'), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      importErrorEl.textContent = (data && data.detail) || '创建下载任务失败';
      uploadBtn.disabled = false;
      updateUploadBtnLabel();
      return;
    }
    urlInput.value = '';
    if (cookiesFileInput) cookiesFileInput.value = '';
    await refreshJobs({ keepSelection: true });
  } catch (err) {
    importErrorEl.textContent = '请求失败：' + (err && err.message ? err.message : String(err));
  }
  uploadBtn.disabled = false;
  isSubmitting = false;
  updateUploadBtnLabel();
}

checkCookiesBtn.addEventListener('click', async () => {
  cookiesResultEl.textContent = '检查中...';
  cookiesResultEl.className = 'cookies-result';
  const form = new FormData();
  form.append('browser', cookiesBrowserSelect.value);
  if (cookiesFileInput.files && cookiesFileInput.files[0]) {
    form.append('file', cookiesFileInput.files[0]);
  }
  try {
    const res = await fetch(apiUrl('api/cookies/check'), { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      cookiesResultEl.textContent = (data && data.detail) || '检查失败';
      cookiesResultEl.className = 'cookies-result bad';
      return;
    }
    const bc = data.browser_check || {};
    const parts = [];
    if (bc.valid) {
      parts.push('浏览器: 可用');
      if (bc.profile) parts.push('配置: ' + bc.profile);
      if (bc.warning) parts.push(bc.warning);
    } else {
      parts.push('浏览器: ' + (bc.error || '不可用'));
    }
    if (data.file_check) {
      const fc = data.file_check;
      if (fc.valid) {
        parts.push('Cookies.txt: 有效 (' + (fc.domain_count || 0) + ' 个域名)');
      } else {
        parts.push('Cookies.txt: ' + (fc.error || '无效'));
      }
    }
    cookiesResultEl.textContent = parts.join(' | ');
    cookiesResultEl.className = 'cookies-result ' + (bc.valid ? 'ok' : 'bad');
  } catch (err) {
    cookiesResultEl.textContent = '请求失败：' + (err && err.message ? err.message : String(err));
    cookiesResultEl.className = 'cookies-result bad';
  }
});

function confirmLeaveUnsavedChanges() {
  if (!editorDirty) return true;
  return window.confirm('有未保存的字幕修改，离开后将丢失。确定离开吗？');
}

newTaskBtn.addEventListener('click', () => { if (confirmLeaveUnsavedChanges()) showImportView({ clearDraft: true }); });
openNewBtn.addEventListener('click', () => { if (confirmLeaveUnsavedChanges()) showImportView({ clearDraft: true }); });
backFromProcessingBtn.addEventListener('click', () => { if (confirmLeaveUnsavedChanges()) showImportView({ clearDraft: true }); });
refreshJobsBtn.addEventListener('click', () => refreshJobs());
backToTasksBtn.addEventListener('click', () => { if (confirmLeaveUnsavedChanges()) showImportView({ clearDraft: true }); });
openSettingsBtn.addEventListener('click', () => { openSettings(); loadHotwordsGlossary(); });
closeSettingsBtn.addEventListener('click', closeSettings);
openTranslateBtn.addEventListener('click', openTranslate);
closeTranslateBtn.addEventListener('click', closeTranslate);
openProofreadBtn.addEventListener('click', openProofreadModal);
closeProofreadBtn.addEventListener('click', () => { stopProofreadPolling(); proofreadModal.classList.add('is-hidden'); });
proofreadRunBtn.addEventListener('click', runProofread);
proofreadApplyBtn.addEventListener('click', applyProofread);
proofreadTermsAllEl.addEventListener('change', () => toggleProofreadGroup('term'));
proofreadTyposAllEl.addEventListener('change', () => toggleProofreadGroup('typo'));
llmProfileAddBtn.addEventListener('click', () => showLlmProfileEditor(null));
llmProfileSaveBtn.addEventListener('click', saveLlmProfile);
llmProfileCancelBtn.addEventListener('click', hideLlmProfileEditor);
llmProfileTestBtn.addEventListener('click', testLlmProfile);
hotwordsGlossaryEditBtn.addEventListener('click', () => {
  hotwordsGlossaryTextEl.value = (hotwordsGlossary || []).join(' ');
  hotwordsGlossaryEditorEl.classList.remove('is-hidden');
});
hotwordsGlossaryCancelBtn.addEventListener('click', () => hotwordsGlossaryEditorEl.classList.add('is-hidden'));
hotwordsGlossarySaveBtn.addEventListener('click', async () => {
  const terms = hotwordsGlossaryTextEl.value.split(/[\\s,，;；\\n]+/).map((s) => s.trim()).filter(Boolean);
  try {
    const res = await fetch(apiUrl('api/hotwords'), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ terms })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '保存失败');
    hotwordsGlossary = data.terms || [];
    renderHotwordsGlossary();
    hotwordsGlossaryEditorEl.classList.add('is-hidden');
  } catch (err) {
    alert('保存失败：' + (err.message || err));
  }
});

async function loadHotwordsGlossary() {
  try {
    const res = await fetch(apiUrl('api/hotwords'));
    if (!res.ok) return;
    const data = await res.json();
    hotwordsGlossary = data.terms || [];
    renderHotwordsGlossary();
  } catch (err) { /* ignore */ }
}

function renderHotwordsGlossary() {
  if (!hotwordsGlossary || !hotwordsGlossary.length) {
    hotwordsGlossaryViewEl.innerHTML = '<div class="meta">词表为空。可手动添加，或在校对应用术语时自动积累。</div>';
    return;
  }
  hotwordsGlossaryViewEl.innerHTML = `<div class="llm-profile-item"><div class="llm-profile-item-info"><div class="llm-profile-item-meta" style="white-space:normal">${hotwordsGlossary.map(escapeHtml).join(' · ')}</div></div><div class="llm-profile-item-meta">${hotwordsGlossary.length} 个热词</div></div>`;
}
openClipsBtn.addEventListener('click', openClips);
closeClipsBtn.addEventListener('click', closeClips);
settingsModal.addEventListener('click', (event) => {
  if (event.target === settingsModal) closeSettings();
});
translateModal.addEventListener('click', (event) => {
  if (event.target === translateModal) closeTranslate();
});
clipsModal.addEventListener('click', (event) => {
  if (event.target === clipsModal) closeClips();
});
deleteCurrentBtn.addEventListener('click', async () => {
  if (currentJob) await deleteJob(currentJob.id);
});

cancelCurrentBtn.addEventListener('click', async () => {
  if (currentJob && window.confirm('确定取消当前任务？已完成的转录/翻译产物会保留。')) {
    await cancelJobById(currentJob.id);
  }
});

jobListEl.addEventListener('click', async (event) => {
  if (event.target.closest('a')) {
    event.stopPropagation();
    return;
  }
  const deleteButton = event.target.closest('[data-delete-id]');
  if (deleteButton) {
    event.stopPropagation();
    await deleteJob(deleteButton.dataset.deleteId);
    return;
  }
  const cancelButton = event.target.closest('[data-cancel-id]');
  if (cancelButton) {
    event.stopPropagation();
    if (window.confirm('确定取消该任务？已完成的转录/翻译产物会保留。')) {
      await cancelJobById(cancelButton.dataset.cancelId);
    }
    return;
  }
  const item = event.target.closest('[data-job-id]');
  if (item) await selectJob(item.dataset.jobId);
});

function defaultPendingParams() {
  return {
    prompt: storedDefaults.prompt,
    speakerCount: storedDefaults.speakerCount || '',
    diarizationBackend: storedDefaults.backend || 'auto'
  };
}

function pendingSummary(item) {
  const parts = [];
  if (item.speakerCount) parts.push('speakers ' + item.speakerCount);
  if (item.hotwords) {
    const count = item.hotwords.split(/[\\s,，;；]+/).filter(Boolean).length;
    if (count) parts.push('热词 ' + count);
  }
  parts.push(diarizationBackendLabel(item.diarizationBackend || 'auto'));
  return parts.join(' · ');
}

function renderPendingList() {
  if (!pendingUploads.length) {
    pendingListEl.innerHTML = '';
    return;
  }
  pendingListEl.innerHTML = pendingUploads.map((item) => `
    <div class="pending-item" data-id="${item.id}">
      <div class="pending-item-head">
        <span class="pending-item-name">${escapeHtml(item.file.name)}</span>
        <span class="pending-item-summary">${escapeHtml(pendingSummary(item))}</span>
        <button class="pending-remove" type="button" data-action="remove" title="移除">✕</button>
      </div>
      <div class="pending-item-body">
        <div>
          <label>推理 Prompt</label>
          <textarea class="pending-prompt prompt-input" rows="2">${escapeHtml(item.prompt)}</textarea>
        </div>
        <div class="params-row">
          <div>
            <label>说话人数</label>
            <input class="pending-speakers" type="number" min="0" max="10" step="1" value="${escapeHtml(String(item.speakerCount || ''))}" />
          </div>
          <div>
            <label>后端</label>
            <select class="pending-diarization">
              <option value="none"${(item.diarizationBackend || 'none') === 'none' ? ' selected' : ''}>不分说话人</option>
              <option value="auto"${item.diarizationBackend === 'auto' ? ' selected' : ''}>auto（推荐）</option>
              <option value="pyannote"${item.diarizationBackend === 'pyannote' ? ' selected' : ''}>pyannote</option>
              <option value="cluster"${item.diarizationBackend === 'cluster' ? ' selected' : ''}>cluster</option>
            </select>
          </div>
        </div>
      </div>
    </div>`).join('');
  pendingListEl.querySelectorAll('textarea.pending-prompt').forEach(el => autoGrow(el));
}

function updatePendingSummary(id) {
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  const row = pendingListEl.querySelector('.pending-item[data-id="' + id + '"] .pending-item-summary');
  if (row) row.textContent = pendingSummary(item);
}

function diarizationBackendLabel(value) {
  const labels = {
    auto: '说话人 auto',
    pyannote: '说话人 pyannote',
    cluster: '说话人 cluster',
    none: '不标说话人'
  };
  return labels[value] || ('说话人 ' + value);
}

function updateUploadBtnLabel() {
  if (rerunDraftJob) {
    uploadBtn.textContent = '开始重跑';
  } else if (document.querySelector('#urlTab.active')) {
    uploadBtn.textContent = '下载并转写';
  } else if (pendingUploads.length) {
    uploadBtn.textContent = '开始转写 ' + pendingUploads.length + ' 个任务';
  } else {
    uploadBtn.textContent = '全部开始转写';
  }
}

function addPendingFiles(fileList) {
  for (const file of fileList) {
    pendingUploads.push(Object.assign({ id: ++pendingIdCounter, file, hotwords: uploadHotwordsInput.value.trim() }, defaultPendingParams()));
  }
  renderPendingList();
  updateUploadBtnLabel();
}

fileInput.addEventListener('change', () => {
  if (rerunDraftJob) resetImportMode();
  if (fileInput.files && fileInput.files.length) {
    addPendingFiles(fileInput.files);
  }
  fileInput.value = '';
});

pendingListEl.addEventListener('click', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const action = event.target.dataset.action;
  if (action === 'remove') {
    pendingUploads = pendingUploads.filter((p) => p.id !== id);
    renderPendingList();
    updateUploadBtnLabel();
  }
});

pendingListEl.addEventListener('input', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  if (event.target.classList.contains('pending-prompt')) { item.prompt = event.target.value; autoGrow(event.target); }
  else if (event.target.classList.contains('pending-speakers')) { item.speakerCount = event.target.value; updatePendingSummary(id); }
});

function autoGrow(el) {
  el.style.height = '40px';
  el.style.height = el.scrollHeight + 'px';
}
document.querySelectorAll('textarea.prompt-input').forEach(el => {
  el.addEventListener('input', () => autoGrow(el));
  autoGrow(el);
});

pendingListEl.addEventListener('change', (event) => {
  const row = event.target.closest('.pending-item');
  if (!row) return;
  const id = Number(row.dataset.id);
  const item = pendingUploads.find((p) => p.id === id);
  if (!item) return;
  if (event.target.classList.contains('pending-diarization')) { item.diarizationBackend = event.target.value; updatePendingSummary(id); }
});

saveBtn.addEventListener('click', async () => {
  await saveSegments();
});
syncSubtitlesBtn.addEventListener('click', async () => {
  if (!currentJob) return;
  await loadSegments(currentJob.id, { preserveSelection: true, force: true });
});
undoBtn.addEventListener('click', undoEdit);
redoBtn.addEventListener('click', redoEdit);

addSegmentBtn.addEventListener('click', addSegmentAtPlayhead);
deleteSegmentBtn.addEventListener('click', deleteActiveSegment);
shiftSegmentLeftBtn.addEventListener('click', () => nudgeActiveSegment('shift', -0.1));
shiftSegmentRightBtn.addEventListener('click', () => nudgeActiveSegment('shift', 0.1));
nudgeStartLeftBtn.addEventListener('click', () => nudgeActiveSegment('start', -0.1));
nudgeStartRightBtn.addEventListener('click', () => nudgeActiveSegment('start', 0.1));
nudgeEndLeftBtn.addEventListener('click', () => nudgeActiveSegment('end', -0.1));
nudgeEndRightBtn.addEventListener('click', () => nudgeActiveSegment('end', 0.1));
settingsSaveBtn.addEventListener('click', () => { saveSegments(); });
useActiveSegmentBtn.addEventListener('click', useActiveSegmentAsClipRange);
findClipsBtn.addEventListener('click', () => findClipCandidates('model'));
findClipsRulesBtn.addEventListener('click', () => findClipCandidates('rules'));
renderClipBtn.addEventListener('click', renderSelectedClip);
renderClipQueueBtn.addEventListener('click', renderQueuedClips);
translateZhBtn.addEventListener('click', translateCurrentSubtitles);
restoreTranslationBtn.addEventListener('click', restoreSourceSubtitles);
translationReviewListEl.addEventListener('click', (event) => {
  const button = event.target.closest('[data-review-action]');
  if (!button) return;
  const item = button.closest('.translation-review-item');
  if (!item) return;
  const index = Number(item.dataset.index);
  const start = Number(item.dataset.start || 0);
  const key = item.dataset.key || '';
  const action = button.dataset.reviewAction;
  if (action === 'dismiss') {
    if (key) dismissedTranslationReviewItems.add(key);
    renderTranslationReview(currentJob);
    return;
  }
  if (Number.isFinite(index) && index >= 0) setActiveSegment(index, true);
  if (Number.isFinite(start)) preview.currentTime = Math.max(0, start);
  if (action === 'play') preview.play().catch(() => {});
  if (action === 'jump') closeTranslate();
  updateSubtitlePreview();
});
proofreadAlignmentListEl.addEventListener('click', (event) => {
  const button = event.target.closest('[data-alignment-action]');
  if (!button) return;
  const item = button.closest('.translation-review-item');
  if (!item) return;
  const index = Number(item.dataset.index);
  const start = Number(item.dataset.start || 0);
  const key = item.dataset.key || '';
  if (button.dataset.alignmentAction === 'dismiss') {
    if (key) dismissedAlignmentItems.add(key);
    renderProofreadResult(proofreadResult);
    return;
  }
  if (Number.isFinite(index) && index >= 0) setActiveSegment(index, true);
  if (Number.isFinite(start)) preview.currentTime = Math.max(0, start);
  if (button.dataset.alignmentAction === 'play') preview.play().catch(() => {});
  proofreadModal.classList.add('is-hidden');
  updateSubtitlePreview();
});
clipTitleInput.addEventListener('input', () => {
  const clip = activeClip();
  if (!clip) return;
  updateClipFromValues(clip.id, { title: clipTitleInput.value });
});
clipStartInput.addEventListener('change', () => {
  const clip = activeClip();
  if (!clip) {
    updateTimelineClipRange();
    return;
  }
  updateClipFromValues(clip.id, { start: Number(clipStartInput.value || 0), end: Number(clipEndInput.value || 0), seek: true });
});
clipEndInput.addEventListener('change', () => {
  const clip = activeClip();
  if (!clip) {
    updateTimelineClipRange();
    return;
  }
  updateClipFromValues(clip.id, { start: Number(clipStartInput.value || 0), end: Number(clipEndInput.value || 0) });
});
clipDurationInput.addEventListener('change', () => {
  const clip = activeClip();
  if (!clip) return;
  const duration = Math.max(0.25, Number(clipDurationInput.value || 0));
  updateClipFromValues(clip.id, { start: clip.start, end: clip.start + duration });
});
clipMoveBackBtn.addEventListener('click', () => nudgeActiveClip({ shift: -1 }));
clipMoveForwardBtn.addEventListener('click', () => nudgeActiveClip({ shift: 1 }));
clipStartEarlierBtn.addEventListener('click', () => nudgeActiveClip({ startDelta: -0.5 }));
clipEndLaterBtn.addEventListener('click', () => nudgeActiveClip({ endDelta: 0.5 }));

renderBtn.addEventListener('click', async () => {
  if (!currentJob || !ffmpegAvailable) return;
  const saved = await saveSegments();
  if (!saved) return;
  const style = collectSubtitleStyle();
  currentJob = { ...currentJob, status: 'rendering', progress: RENDER_PROGRESS_BASE, error: null };
  renderCurrentJob(currentJob, { skipSegments: true });
  const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/render`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ style })
  });
  const data = await res.json();
  if (!res.ok) {
    currentJob = { ...currentJob, status: 'waiting_review', progress: 0.95, error: data.detail || '烧录失败' };
    renderCurrentJob(currentJob, { skipSegments: true });
  }
  else {
    currentJob = data;
    renderCurrentJob(data, { skipSegments: true });
    await refreshJobs({ keepSelection: true });
  }
});

rerunBtn.addEventListener('click', () => {
  if (currentJob) showRerunDraft(currentJob);
});

preview.addEventListener('timeupdate', scheduleActiveSegmentSync);
preview.addEventListener('play', () => {
  timelineFollowHoldUntil = 0;
  syncMaskPreviewPlayback();
});
preview.addEventListener('seeked', () => scheduleActiveSegmentSync(true));
preview.addEventListener('pause', syncMaskPreviewPlayback);
preview.addEventListener('seeking', syncMaskPreviewTime);
preview.addEventListener('seeked', syncMaskPreviewTime);
preview.addEventListener('ratechange', syncMaskPreviewPlaybackRate);
preview.addEventListener('loadedmetadata', () => {
  fitVideoStageToMedia();
  renderTimeline(collectSegments());
  scheduleActiveSegmentSync(true);
  syncMaskPreviewPlaybackRate();
});
timelineScroll.addEventListener('pointerdown', (event) => {
  if (!currentJob || event.target.closest('.timeline-segment, .timeline-clip-range')) return;
  event.preventDefault();
  timelineDragging = true;
  timelineScroll.setPointerCapture(event.pointerId);
  seekTimelineFromPointer(event);
});
timelineClipRange.addEventListener('pointerdown', onClipRangePointerDown);
timelineScroll.addEventListener('pointermove', (event) => {
  if (!timelineDragging) return;
  event.preventDefault();
  updateTimelineEdgeAutoScroll(event, 'seek');
  seekTimelineFromPointer(event);
});
timelineScroll.addEventListener('pointerup', (event) => {
  if (!timelineDragging) return;
  event.preventDefault();
  seekTimelineFromPointer(event);
  timelineDragging = false;
  timelineFollowHoldUntil = Number.POSITIVE_INFINITY;
  stopTimelineEdgeAutoScroll();
  hideTimelineGuide();
  try {
    timelineScroll.releasePointerCapture(event.pointerId);
  } catch (err) {}
});
timelineScroll.addEventListener('pointercancel', () => {
  timelineDragging = false;
  timelineFollowHoldUntil = Number.POSITIVE_INFINITY;
  stopTimelineEdgeAutoScroll();
  hideTimelineGuide();
});
timelineScroll.addEventListener('dragstart', (event) => event.preventDefault());
timelineScroll.addEventListener('scroll', scheduleVisibleTimelineRender);
if (tableWrap) tableWrap.addEventListener('scroll', scheduleVisibleSegmentRowsRender);
window.addEventListener('resize', () => {
  scheduleLayoutFit();
  renderTimeline(collectSegments());
  renderVisibleSegmentRows();
});
if ('ResizeObserver' in window) {
  const layoutObserver = new ResizeObserver(scheduleLayoutFit);
  for (const element of [videoShell, document.querySelector('.content'), document.querySelector('.editor-grid')]) {
    if (element) layoutObserver.observe(element);
  }
}
tbody.addEventListener('input', (event) => {
  const tr = event.target.closest('tr[data-index]');
  // 首次修改时推快照（保存改之前的状态），用于撤销文本编辑
  if (!editorDirty) pushUndoSnapshot();
  updateCachedSegmentFromRow(tr);
  markEditorDirty();
  if (event.target.classList.contains('text')) {
    resizeSegmentTextarea(event.target, tr && tr.classList.contains('active'));
    scheduleVisibleTimelineRender();
  }
  if (event.target.classList.contains('start') || event.target.classList.contains('end')) {
    renderTimeline(collectSegments());
    scheduleActiveSegmentSync(true);
  }
  else {
    if (event.target.classList.contains('speaker')) {
      renderSpeakerMap(collectSegments());
      renderTimeline(collectSegments());
      refreshSpeakerDots();
    }
    updateSubtitlePreview();
  }
});
tbody.addEventListener('change', (event) => {
  // change 可能不经过 input（自动填充等）：先把变更前状态推入撤销栈再更新缓存
  if (!editorDirty) pushUndoSnapshot();
  updateCachedSegmentFromRow(event.target.closest('tr[data-index]'));
  markEditorDirty();
});
tbody.addEventListener('click', (event) => {
  const addAboveButton = event.target.closest('.add-row-above');
  const addBelowButton = event.target.closest('.add-row-below');
  const deleteButton = event.target.closest('.delete-row');
  const mergeButton = event.target.closest('.merge-row-below');
  if (!addAboveButton && !addBelowButton && !deleteButton && !mergeButton) return;
  event.preventDefault();
  event.stopPropagation();
  const tr = event.target.closest('tr');
  if (!tr) return;
  const index = Number(tr.dataset.index);
  if (addAboveButton) addSegmentAroundIndex(index, 'above');
  else if (addBelowButton) addSegmentAroundIndex(index, 'below');
  else if (mergeButton) mergeSegmentWithNext(index);
  else deleteSegmentAtIndex(index);
});
tbody.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || !(event.ctrlKey || event.metaKey)) return;
  const textarea = event.target.closest('textarea.text');
  if (!textarea) return;
  event.preventDefault();
  const tr = textarea.closest('tr');
  if (!tr) return;
  splitSegmentAtCursor(Number(tr.dataset.index), textarea);
});
speakerMapEl.addEventListener('input', (event) => {
  if (event.target.classList.contains('speaker-color')) {
    // 调色盘:记录该说话人的自定义颜色,联动行内圆点/预览/烧录。
    markEditorDirtyWithUndo();
    const speaker = event.target.dataset.speaker || '';
    if (speaker) speakerColorOverrides[speaker] = event.target.value;
    refreshSpeakerDots();
    const dot = event.target.closest('.speaker-map-row')?.querySelector('.speaker-dot');
    if (dot) dot.style.background = event.target.value;
    updateSubtitlePreview();
    return;
  }
  markEditorDirtyWithUndo();
  syncSpeakerNameInputs();
  updateSubtitlePreview();
});
tbody.addEventListener('focusin', (event) => {
  const tr = event.target.closest('tr');
  if (!tr) return;
  const index = Number(tr.dataset.index);
  seekPreviewToSegment(index);
  setActiveSegment(index, false);
  updateTimelinePlayhead();
  resizeSegmentRow(tr, true);
  updateSubtitlePreview();
});
for (const id of ['fontName', 'primaryColor', 'outlineColor', 'fontSize', 'marginV', 'showSpeaker', 'speakerColors', 'maskEnabled', 'maskMode', 'maskHeight', 'maskMarginV', 'maskBlur', 'maskOpacity']) {
  document.querySelector('#' + id).addEventListener('input', () => {
    markEditorDirtyWithUndo();
    updateSubtitlePreview();
    if (id === 'speakerColors') {
      renderSpeakerMap(collectSegments());
      refreshSpeakerDots();
    }
  });
  document.querySelector('#' + id).addEventListener('change', () => {
    markEditorDirtyWithUndo();
    updateSubtitlePreview();
    if (id === 'speakerColors') {
      renderSpeakerMap(collectSegments());
      refreshSpeakerDots();
    }
  });
}

async function refreshJobs(options = {}) {
  const res = await fetch(apiUrl('api/jobs'), { cache: 'no-store' });
  if (!res.ok) return;
  const data = await res.json();
  jobs = data.jobs || [];
  renderJobList();
  if (currentJob) {
    const fresh = jobs.find((job) => job.id === currentJob.id);
    if (fresh) {
      const wasEditable = EDIT_STATES.has(currentJob.status);
      currentJob = fresh;
      if (options.background && wasEditable && EDIT_STATES.has(fresh.status)) {
        updateEditorChrome(fresh);
      } else {
        renderCurrentJob(fresh, { skipSegments: options.skipSegments || editorDirty });
      }
    } else {
      currentJob = null;
      showImportView();
    }
  } else if (!options.keepSelection && jobs.length && options.selectLatest) {
    await selectJob(jobs[0].id);
  }
  ensurePolling();
}

function renderJobList() {
  jobCountEl.textContent = jobs.length + ' 个任务';
  if (!jobs.length) {
    jobListEl.innerHTML = '<div class="meta" style="padding:10px">还没有任务</div>';
    return;
  }
  jobListEl.innerHTML = jobs.map((job) => {
    const active = currentJob && currentJob.id === job.id ? ' active' : '';
    const canDelete = true;
    const percent = Math.round((job.progress || 0) * 100);
    const warning = truncationWarning(job);
    const sourceBadge = job.source === 'url'
      ? '<span class="source-badge url">链接</span>'
      : '<span class="source-badge">上传</span>';
    const dlInfo = (job.status === 'downloading' && job.download_info)
      ? `<div class="meta">下载: ${job.download_info.percent || 0}%${job.download_info.speed ? ' | ' + escapeHtml(job.download_info.speed) : ''}${job.download_info.eta ? ' | ETA ' + escapeHtml(job.download_info.eta) : ''}</div>`
      : '';
    const outputKinds = EDIT_STATES.has(job.status)
      ? [['srt', 'SRT'], ['ass', 'ASS'], ['transcript', 'TXT']].concat(job.status === 'done' ? [['mp4', 'MP4']] : [])
      : [];
    const outputLinks = outputKinds.length
      ? `<div class="task-output-row"><span class="task-output-label">OUTPUT</span>${outputKinds.map(([kind, label]) => `<a href="${apiUrl(`api/jobs/${job.id}/download?kind=${kind}`)}" target="_blank" rel="noreferrer">${label}</a>`).join('<span class="meta">·</span>')}</div>`
      : '';
    return `
      <div class="task-item${active}" data-job-id="${escapeHtml(job.id)}">
        <div class="task-row">
          <div class="task-name">${sourceBadge}${escapeHtml(job.media_name || 'input.media')}</div>
          <span class="${statusClass(job.status)}">${statusLabel(job.status)}</span>
        </div>
        <div class="task-id meta">${escapeHtml(job.id)}</div>
        ${dlInfo}
        ${job.status === 'downloading' ? '' : `<div class="meta">${escapeHtml(tokenUsageSummary(job))}</div>`}
        ${warning ? `<div class="warning">${escapeHtml(warning)}</div>` : ''}
        <div class="task-foot">
          <div class="progress task-progress"><div class="bar" style="width:${percent}%"></div></div>
          ${RUNNING_STATES.has(job.status) ? `<button class="small ghost" data-cancel-id="${escapeHtml(job.id)}">取消</button>` : ''}
          ${canDelete ? `<button class="small ghost" data-delete-id="${escapeHtml(job.id)}">${RUNNING_STATES.has(job.status) ? '取消并删除' : '删除'}</button>` : ''}
        </div>
        ${outputLinks}
      </div>`;
  }).join('');
}

async function selectJob(jobId) {
  const local = jobs.find((job) => job.id === jobId);
  currentJob = local || currentJob;
  renderJobList();
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { cache: 'no-store' });
  if (!res.ok) {
    await refreshJobs();
    return;
  }
  currentJob = await res.json();
  openJobEvents(currentJob.id);
  renderCurrentJob(currentJob);
}

function renderCurrentJob(job, options = {}) {
  if (job.id !== translationReviewJobId) {
    translationReviewJobId = job.id;
    dismissedTranslationReviewItems = new Set();
    dismissedAlignmentItems = new Set();
  }
  ensureClipQueueForJob();
  renderJobList();
  if (EDIT_STATES.has(job.status)) showEditor(job, options);
  else showProcessing(job);
}

function showImportView(options = {}) {
  if (options.clearDraft !== false) resetImportMode();
  currentJob = null;
  stopSubtitleSyncPolling();
  closeJobEvents();
  cachedSegments = null;
  cachedTimelineSegments = [];
  resetUndoHistory();
  ensureClipQueueForJob();
  closeSettings();
  closeTranslate();
  closeClips();
  setEditorDirty(false);
  fileInput.value = '';
  const uploadTabBtn = document.querySelector('.tab-btn[data-tab="upload"]');
  if (uploadTabBtn) uploadTabBtn.click();
  if (!options.preserveError) importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

function resetImportMode() {
  rerunDraftJob = null;
  importTitleEl.textContent = '任务管理';
  rerunSourceEl.textContent = '';
  fileInput.disabled = false;
  if (rerunParamsEl) rerunParamsEl.style.display = 'none';
  pendingUploads = [];
  renderPendingList();
  if (urlInput) urlInput.value = '';
  if (cookiesFileInput) cookiesFileInput.value = '';
  if (cookiesResultEl) {
    cookiesResultEl.textContent = '';
    cookiesResultEl.className = 'cookies-result';
  }
  updateUploadBtnLabel();
}

function showProcessingPlaceholder(name) {
  currentJob = null;
  closeSettings();
  closeTranslate();
  closeClips();
  processTitleEl.textContent = '创建任务';
  processNameEl.textContent = name;
  processMetaEl.textContent = '上传媒体并准备转写';
  processBarEl.classList.remove('indeterminate');
  processBarEl.style.width = '2%';
  processErrorEl.textContent = '';
  setVisible(processingView);
}

function showProcessing(job) {
  if (job.status === 'failed') {
    processTitleEl.textContent = '任务失败';
  } else {
    processTitleEl.textContent = statusLabel(job.status);
  }
  processNameEl.textContent = job.media_name || 'input.media';
  let meta;
  if (job.status === 'downloading') {
    const dl = job.download_info || {};
    const parts = ['下载 ' + (dl.percent || 0) + '%'];
    if (dl.speed) parts.push(dl.speed);
    if (dl.eta) parts.push('ETA ' + dl.eta);
    meta = parts.join(' | ');
  } else {
    meta = jobSummary(job);
  }
  processMetaEl.textContent = meta;
  const rawPercent = Math.round((job.progress || 0) * 100);
  const staticPhases = new Set(['loading_model', 'postprocessing', 'labeling_speakers']);
  const isIndeterminate = (job.status === 'transcribing' && rawPercent <= 12) || staticPhases.has(job.status);
  processBarEl.classList.toggle('indeterminate', isIndeterminate);
  processBarEl.style.width = `${Math.max(rawPercent, 5)}%`;
  processErrorEl.textContent = job.error || truncationWarning(job);
  updateTranslateProgress(job);
  deleteCurrentBtn.disabled = false;
  cancelCurrentBtn.classList.toggle('is-hidden', !RUNNING_STATES.has(job.status));
  setVisible(processingView);
}

async function showEditor(job, options = {}) {
  applySubtitleStyle(job.subtitle_style || {});
  updateEditorChrome(job);
  setVisible(workbench);
  closeSettings();
  closeTranslate();
  closeClips();
  const mediaUrl = apiUrl(`api/jobs/${job.id}/media`);
  if (preview.dataset.jobId !== job.id) {
    preview.dataset.jobId = job.id;
    setPreviewSource(mediaUrl);
    resetVideoStage();
  }
  if (!options.skipSegments) await loadSegments(job.id);
  fitVideoStageToMedia();
  startSubtitleSyncPolling();
}

function updateEditorChrome(job) {
  selectedNameEl.textContent = job.media_name || 'input.media';
  taskStatusEl.textContent = statusLabel(job.status);
  taskStatusEl.className = statusClass(job.status);
  taskUsageEl.textContent = tokenUsageSummary(job);
  taskParamsEl.textContent = parameterSummary(job);
  updateRenderProgress(job);
  updateTranslateProgress(job);
  if (job.error) setTaskNotice(job.error, 'error');
  else if (truncationWarning(job)) setTaskNotice('可能截断，建议提高输出 tokens 后重新转写。', 'warning');
  else setTaskNotice('', '');
  updateRenderAction(job);
  updateTranslateAction();
  updateProofreadAction();
  updateClipActions();
  updateRerunAction(job);
  if (syncSubtitlesBtn) syncSubtitlesBtn.disabled = editorDirty || !EDIT_STATES.has(job.status);
  setSaveState(editorDirty ? 'dirty' : 'saved', editorDirty ? '有未保存修改' : '已保存');
}

function updateRenderAction(job) {
  const isRendering = job && job.status === 'rendering';
  if (!runtimeChecked) {
    renderBtn.disabled = true;
    renderBtn.textContent = '检测 FFmpeg...';
  } else {
    renderBtn.disabled = !ffmpegAvailable || isRendering;
    renderBtn.textContent = isRendering ? '烧录中...' : ffmpegAvailable ? '烧录视频' : 'FFmpeg 不可用';
  }
}

function updateRenderProgress(job) {
  const isRendering = job.status === 'rendering';
  const showProgress = isRendering || job.status === 'done';
  renderProgressMetaEl.classList.toggle('is-hidden', !showProgress);
  renderProgressEl.classList.toggle('is-hidden', !showProgress);
  const renderRatio = job.status === 'done' ? 1 : Math.max(0, Math.min(1, ((job.progress || RENDER_PROGRESS_BASE) - RENDER_PROGRESS_BASE) / RENDER_PROGRESS_SPAN));
  const percent = Math.round(renderRatio * 100);
  renderProgressBarEl.style.width = `${percent}%`;
  renderProgressTextEl.textContent = `${percent}%`;
}

function translationProgressSummary(job) {
  const translation = (job && job.translation) || {};
  const done = Number(translation.done || 0);
  const total = Number(translation.total || 0);
  const percent = translation.percent == null
    ? (total > 0 ? Math.round(done * 1000 / total) / 10 : 0)
    : Number(translation.percent || 0);
  const elapsed = Number(translation.elapsed_sec || 0);
  const elapsedText = elapsed > 0 ? ' · ' + formatDuration(elapsed) : '';
  if (total > 0) return `翻译 ${done}/${total} (${Math.round(percent)}%)${elapsedText}`;
  return `翻译中${elapsedText}`;
}

function updateTranslateProgress(job) {
  const translation = (job && job.translation) || {};
  const showProgress = job && (job.status === 'translating' || translation.in_progress || translation.percent != null);
  translateProgressMetaEl.classList.toggle('is-hidden', !showProgress);
  translateProgressEl.classList.toggle('is-hidden', !showProgress);
  if (!showProgress) return;
  const done = Number(translation.done || 0);
  const total = Number(translation.total || 0);
  const percent = Math.max(0, Math.min(100, translation.percent == null
    ? (total > 0 ? done * 100 / total : 0)
    : Number(translation.percent || 0)));
  translateProgressBarEl.style.width = `${percent}%`;
  translateProgressTextEl.textContent = total > 0
    ? `${done}/${total} (${Math.round(percent)}%)`
    : `${Math.round(percent)}%`;
  if (job.status === 'translating') translateStatusEl.textContent = translationProgressSummary(job);
}

function renderTranslationReview(jobOrPayload) {
  const translation = translationPayload(jobOrPayload);
  const issues = Array.isArray(translation.validation_issues) ? translation.validation_issues : [];
  const skips = Array.isArray(translation.pretranslation_skips) ? translation.pretranslation_skips : [];
  const reviewItems = [
    ...issues.map((item) => ({ ...item, severity: 'warning', group: 'issue' })),
    ...skips.map((item) => ({ ...item, severity: 'info', group: 'skip' })),
  ].filter((item) => !dismissedTranslationReviewItems.has(reviewItemKey(item)));
  const issueCount = Number(translation.validation_issue_count == null ? issues.length : translation.validation_issue_count);
  const skipCount = Number(translation.pretranslation_skip_count == null ? skips.length : translation.pretranslation_skip_count);
  translationReviewEl.classList.toggle('is-hidden', !issueCount && !skipCount);
  if (!issueCount && !skipCount) {
    translationReviewMetaEl.textContent = '';
    translationReviewListEl.innerHTML = '';
    return;
  }
  translationReviewMetaEl.textContent = `${issueCount} 条可疑 · ${skipCount} 条自动跳过`;
  if (!reviewItems.length) {
    translationReviewListEl.innerHTML = '<div class="meta">当前显示项都已标记处理。</div>';
    return;
  }
  translationReviewListEl.innerHTML = reviewItems.map((item) => {
    const index = Number(item.index);
    const start = Number(item.start || 0);
    const key = reviewItemKey(item);
    const title = item.group === 'skip' ? '自动跳过' : '可疑翻译';
    const detail = reviewIssueLabel(item);
    const text = item.text || item.source_text || '';
    return `
      <div class="translation-review-item ${item.severity}" data-index="${Number.isFinite(index) ? index : -1}" data-start="${Number.isFinite(start) ? start : 0}" data-key="${escapeHtml(key)}">
        <div class="translation-review-item-head">
          <span>${title} · #${Number.isFinite(index) ? index + 1 : '?'} · ${formatTimelineTime(start)}</span>
          <strong>${escapeHtml(detail)}</strong>
        </div>
        <div class="translation-review-text">${escapeHtml(text || '(空)')}</div>
        <div class="translation-review-actions">
          <button class="ghost small" type="button" data-review-action="jump">跳到字幕</button>
          <button class="ghost small" type="button" data-review-action="play">回看原片</button>
          <button class="primary small" type="button" data-review-action="dismiss">标为已处理</button>
        </div>
      </div>`;
  }).join('');
}

function translationPayload(jobOrPayload) {
  if (!jobOrPayload) return {};
  if (jobOrPayload.translation) return jobOrPayload.translation || {};
  return jobOrPayload;
}

function reviewItemKey(item) {
  return [item.group || item.type || 'review', item.id || '', item.index == null ? '' : item.index, item.reason || ''].join(':');
}

function reviewIssueLabel(item) {
  const labels = {
    pretranslation_skip: item.reason ? skipReasonLabel(item.reason) : '跳过翻译',
    count_mismatch: '数量不一致',
    model_artifact: '模型副产物',
    json_fragment: 'JSON 残片',
    suspicious_expansion: '译文异常变长',
    empty_translation: '空译文',
  };
  return labels[item.type] || item.reason || item.type || '待检查';
}

function skipReasonLabel(reason) {
  const labels = {
    empty: '空字幕',
    known_transcript_noise: '已知转写噪声',
    bracketed_effect: '括号音效',
    too_short: '过短',
    short_code_or_noise: '短代码/噪声',
    repeated_chant: '重复 chant',
    filler_noise: '语气噪声',
  };
  return labels[reason] || reason;
}

function setTaskNotice(message, kind) {
  taskNoticeEl.textContent = message || '';
  taskNoticeEl.className = 'task-notice ' + (kind || '');
  taskNoticeEl.classList.toggle('is-hidden', !message);
}

function updateRerunAction(job) {
  rerunBtn.disabled = RUNNING_STATES.has(job.status);
  rerunBtn.textContent = '重新转写';
}

function showRerunDraft(job) {
  const inference = job.inference || {};
  rerunDraftJob = job;
  currentJob = null;
  importTitleEl.textContent = '重跑转写';
  fileInput.value = '';
  fileInput.disabled = true;
  rerunSourceEl.textContent = '来源媒体：' + (job.media_name || 'input.media');
  rerunPromptInput.value = inference.prompt || '';
  rerunSpeakerCountInput.value = job.speaker_count || '';
  rerunDiarizationSelect.value = job.diarization_backend || 'auto';
  rerunParamsEl.style.display = '';
  uploadBtn.textContent = '开始重跑';
  importErrorEl.textContent = '';
  setVisible(importView);
  renderJobList();
}

async function startRerunDraft() {
  if (!rerunDraftJob) return;
  const source = rerunDraftJob;
  const payload = {
    prompt: rerunPromptInput.value,
  };
  if (normalizedSpeakerCount(rerunSpeakerCountInput.value)) payload.speaker_count = Number(normalizedSpeakerCount(rerunSpeakerCountInput.value));
  payload.diarization_backend = rerunDiarizationSelect.value || 'auto';
  uploadBtn.disabled = true;
  rerunParamsEl.style.display = 'none';
  importErrorEl.textContent = '';
  showProcessingPlaceholder(source.media_name || 'input.media');
  const res = await fetch(apiUrl(`api/jobs/${source.id}/rerun`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  uploadBtn.disabled = false;
  isSubmitting = false;
  if (!res.ok) {
    importErrorEl.textContent = data.detail || '重跑失败';
    showImportView({ clearDraft: false, preserveError: true });
    return;
  }
  resetImportMode();
  currentJob = data;
  await refreshJobs({ keepSelection: true });
  await selectJob(data.id);
}

function setVisible(view) {
  importView.classList.toggle('is-hidden', view !== importView);
  processingView.classList.toggle('is-hidden', view !== processingView);
  workbench.classList.toggle('is-hidden', view !== workbench);
}

async function cancelJobById(jobId) {
  try {
    const res = await fetch(apiUrl(`api/jobs/${jobId}/cancel`), { method: 'POST' });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '取消失败', 'error');
      return;
    }
    applyJobUpdate(data);
    setTaskNotice('已请求取消，正在等任务停下…', '');
  } catch (err) {
    setTaskNotice('取消失败：' + (err.message || err), 'error');
  }
}

async function deleteJob(jobId) {
  // 删除不可恢复（整个 job 目录含媒体/字幕/剪辑），必须确认；
  // 删的是当前打开的任务且带未保存修改时给出更重的警告。
  const deletingCurrent = currentJob && currentJob.id === jobId;
  const message = deletingCurrent && editorDirty
    ? '确定删除该任务？任务产物与未保存的字幕修改都会一并丢失。'
    : '确定删除该任务？媒体、字幕与剪辑产物会一并删除，不可恢复。';
  if (!window.confirm(message)) return;
  const res = await fetch(apiUrl(`api/jobs/${jobId}`), { method: 'DELETE' });
  if (!res.ok) return;
  if (currentJob && currentJob.id === jobId) {
    currentJob = null;
    stopSubtitleSyncPolling();
    preview.removeAttribute('src');
    maskPreviewVideo.removeAttribute('src');
    preview.removeAttribute('data-job-id');
    preview.load();
    maskPreviewVideo.load();
    tbody.innerHTML = '';
    cachedSegments = null;
    cachedTimelineSegments = [];
    clipListEl.innerHTML = '';
    clipStatusEl.textContent = '';
    setEditorDirty(false);
    showImportView();
  }
  await refreshJobs({ keepSelection: true });
}

async function saveSegments(force = false) {
  if (!currentJob) return false;
  if (!editorDirty && !force) return true;
  setSaveState('saving', '正在保存...');
  const segments = collectSegments();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments, style: collectSubtitleStyle() })
    });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '保存失败', 'error');
      setSaveState('error', data.detail || '保存失败');
      saveBtn.disabled = false;
      return false;
    }
    setTaskNotice('', '');
    renderSegments(data.segments);
    setEditorDirty(false);
    saveStatusEl.textContent = '已保存';
    saveStatusTimer = setTimeout(() => {
      if (!editorDirty) saveStatusEl.textContent = '已保存';
    }, 1200);
    await selectJob(currentJob.id);
    return true;
  } catch (err) {
    setTaskNotice('保存失败：' + err.message, 'error');
    setSaveState('error', '保存失败');
    saveBtn.disabled = false;
    return false;
  }
}

async function loadSegments(jobId, options = {}) {
  if (!jobId || subtitleSyncInFlight) return false;
  // 兜底：撤销栈还属于另一个任务（未经过任务列表直接切换的路径）时清空，
  // 防止 A 任务的历史快照被撤销进 B 任务的数据
  if (undoJobId && jobId !== undoJobId) resetUndoHistory();
  if (editorDirty && !options.force) return false;
  subtitleSyncInFlight = true;
  try {
    const headers = {};
    if (segmentsEtag && segmentsEtag.jobId === jobId) headers['If-None-Match'] = segmentsEtag.etag;
    const res = await fetch(apiUrl(`api/jobs/${jobId}/segments`), { cache: 'no-store', headers });
    if (res.status === 304) {
      // 服务器侧 segments.json 没变：不解析、不深比较、不重渲染。
      // 不动 editorDirty——force 同步时若带本地编辑，说明服务器没有新东西，脏状态如实保留。
      return true;
    }
    const data = await res.json();
    const segments = data.segments || [];
    const etag = res.headers.get('etag');
    segmentsEtag = etag ? { jobId, etag } : null;
    // 轮询拉到的数据如果和当前缓存一致（条目数+各条起止/文本/speaker 没变），
    // 就跳过 renderSegments 的整体重建，避免播放/编辑期间每 2 秒整表清空重建造成卡顿
    if (options.preserveSelection && cachedSegments && segments.length === cachedSegments.length && segmentsEveryEqual(segments, cachedSegments)) {
      setEditorDirty(false);
      return true;
    }
    const preferredIndex = options.preserveSelection && activeSegmentIndex >= 0 ? activeSegmentIndex : null;
    renderSegments(segments, preferredIndex);
    setEditorDirty(false);
    return true;
  } finally {
    subtitleSyncInFlight = false;
  }
}

function segmentsEveryEqual(incoming, cached) {
  if (!incoming || !cached || incoming.length !== cached.length) return false;
  for (let i = 0; i < incoming.length; i++) {
    const a = incoming[i];
    const b = cached[i];
    if (!a || !b) return false;
    if (Number(a.start) !== Number(b.start) || Number(a.end) !== Number(b.end)) return false;
    if ((a.speaker || '') !== (b.speaker || '')) return false;
    if (String(a.text || '') !== String(b.text || '')) return false;
  }
  return true;
}

function ensurePolling() {
  const shouldPoll = jobs.some((job) => RUNNING_STATES.has(job.status));
  // SSE 已连上时当前任务的状态是实时推的，轮询降频兜底其他任务与断线。
  const interval = jobEventSource ? 4000 : 1500;
  if (shouldPoll && !pollTimer) pollTimer = setInterval(() => refreshJobs({ keepSelection: true, background: true }), interval);
  if (!shouldPoll && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// -------------------------------------------------------------- SSE 实时状态

let jobEventSource = null;

function closeJobEvents() {
  if (jobEventSource) {
    jobEventSource.close();
    jobEventSource = null;
    ensurePolling();
  }
}

function openJobEvents(jobId) {
  closeJobEvents();
  if (!jobId || typeof EventSource === 'undefined') return;
  const source = new EventSource(apiUrl(`api/jobs/${jobId}/events`));
  source.addEventListener('job', (event) => {
    let job;
    try {
      job = JSON.parse(event.data);
    } catch (err) {
      return;
    }
    applyJobUpdate(job);
  });
  // 断线时 EventSource 会自动重连；重连成功后服务器会先推一份快照。
  source.onerror = () => {};
  jobEventSource = source;
}

function applyJobUpdate(job) {
  if (!job || !job.id) return;
  const index = jobs.findIndex((item) => item.id === job.id);
  if (index >= 0) jobs[index] = job;
  else jobs.unshift(job);
  renderJobList();
  ensurePolling();
  if (currentJob && currentJob.id === job.id) {
    const wasEditable = EDIT_STATES.has(currentJob.status);
    currentJob = job;
    if (wasEditable && EDIT_STATES.has(job.status)) {
      updateEditorChrome(job);
    } else {
      renderCurrentJob(job, { skipSegments: editorDirty });
    }
  }
}

function htmlColorToAss(hex) {
  const value = String(hex || '').replace('#', '');
  if (!/^[0-9A-Fa-f]{6}$/.test(value)) return '&H00FFFFFF';
  const r = value.slice(0, 2), g = value.slice(2, 4), b = value.slice(4, 6);
  return ('&H00' + b + g + r).toUpperCase();
}

function assColorToHtml(ass) {
  const match = /^&H[0-9A-Fa-f]{2}([0-9A-Fa-f]{6})$/.exec(String(ass || ''));
  if (!match) return '#ffffff';
  const bgr = match[1];
  return '#' + (bgr.slice(4, 6) + bgr.slice(2, 4) + bgr.slice(0, 2)).toLowerCase();
}

function collectSubtitleStyle() {
  return {
    font_name: document.querySelector('#fontName').value || 'Noto Sans CJK SC',
    font_size: Number(document.querySelector('#fontSize').value || 48),
    margin_v: Number(document.querySelector('#marginV').value || 56),
    primary_color: htmlColorToAss(document.querySelector('#primaryColor').value),
    outline_color: htmlColorToAss(document.querySelector('#outlineColor').value),
    show_speaker: document.querySelector('#showSpeaker').value === 'true',
    speaker_colors: document.querySelector('#speakerColors').value === 'true',
    speaker_names: collectSpeakerNames(),
    speaker_color_overrides: Object.fromEntries(
      Object.entries(speakerColorOverrides).map(([speaker, hex]) => [speaker, htmlColorToAss(hex)])
    ),
    mask_enabled: document.querySelector('#maskEnabled').value === 'true',
    mask_mode: document.querySelector('#maskMode').value || 'blur',
    mask_height: Number(document.querySelector('#maskHeight').value || 120),
    mask_margin_v: Number(document.querySelector('#maskMarginV').value || 0),
    mask_opacity: Number(document.querySelector('#maskOpacity').value || 0.82),
    mask_blur: Number(document.querySelector('#maskBlur').value || 24)
  };
}

function applySubtitleStyle(style) {
  if (!style || editorDirty) return;
  if (style.font_name) {
    const select = document.querySelector('#fontName');
    const value = String(style.font_name);
    if (![...select.options].some((option) => option.value === value)) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    }
    select.value = value;
  }
  if (style.primary_color != null) document.querySelector('#primaryColor').value = assColorToHtml(style.primary_color);
  if (style.outline_color != null) document.querySelector('#outlineColor').value = assColorToHtml(style.outline_color);
  if (style.font_size != null) document.querySelector('#fontSize').value = style.font_size;
  if (style.margin_v != null) document.querySelector('#marginV').value = style.margin_v;
  if (style.show_speaker != null) document.querySelector('#showSpeaker').value = String(!!style.show_speaker);
  if (style.speaker_colors != null) document.querySelector('#speakerColors').value = String(!!style.speaker_colors);
  if (style.mask_enabled != null) document.querySelector('#maskEnabled').value = String(!!style.mask_enabled);
  if (style.mask_mode != null) document.querySelector('#maskMode').value = style.mask_mode === 'bar' ? 'bar' : 'blur';
  if (style.mask_height != null) document.querySelector('#maskHeight').value = style.mask_height;
  if (style.mask_margin_v != null) document.querySelector('#maskMarginV').value = style.mask_margin_v;
  if (style.mask_opacity != null) document.querySelector('#maskOpacity').value = style.mask_opacity;
  if (style.mask_blur != null) document.querySelector('#maskBlur').value = style.mask_blur;
  speakerNameMap = {};
  speakerMapEl.innerHTML = '';
  const names = style.speaker_names || {};
  for (const [speaker, name] of Object.entries(names)) {
    if (String(name).trim()) speakerNameMap[String(speaker)] = String(name).trim();
  }
  speakerColorOverrides = {};
  const overrides = style.speaker_color_overrides || {};
  for (const [speaker, assColor] of Object.entries(overrides)) {
    speakerColorOverrides[String(speaker)] = assColorToHtml(assColor);
  }
}

function collectSpeakerNames() {
  const names = {};
  for (const input of speakerMapEl.querySelectorAll('input.speaker-name[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    const name = input.value.trim();
    if (speaker && name) names[speaker] = name;
  }
  return names;
}

function syncSpeakerNameInputs() {
  for (const input of speakerMapEl.querySelectorAll('input.speaker-name[data-speaker]')) {
    const speaker = input.dataset.speaker || '';
    if (!speaker) continue;
    const name = input.value.trim();
    if (name) speakerNameMap[speaker] = name;
    else delete speakerNameMap[speaker];
  }
}

function renderSpeakerMap(segments) {
  syncSpeakerNameInputs();
  const speakers = [...new Set(segments.map((segment) => segment.speaker).filter(Boolean))].sort();
  if (!speakers.length) {
    speakerMapEl.innerHTML = '<div class="meta">暂无说话人</div>';
    return;
  }
  const useColors = document.querySelector('#speakerColors').value === 'true' && speakers.length > 1;
  speakerMapEl.innerHTML = speakers.map((speaker) => {
    const name = speakerNameMap[speaker] || '';
    const color = speakerColorOf(speaker, speakers);
    const picker = useColors
      ? `<input type="color" class="speaker-color" data-speaker="${escapeHtml(speaker)}" value="${color}" title="该说话人的字幕颜色（点击调整）">`
      : '';
    return `
      <div class="speaker-map-row">
        <div class="speaker-tag"><span class="speaker-dot" style="background:${useColors ? color : 'transparent'}"></span>${escapeHtml(speaker)}</div>
        <input type="text" class="speaker-name" data-speaker="${escapeHtml(speaker)}" value="${escapeHtml(name)}" placeholder="显示名称">
        ${picker}
      </div>`;
  }).join('');
}

function speakerDisplayName(speaker) {
  const names = collectSpeakerNames();
  return names[speaker] || speakerNameMap[speaker] || speaker;
}

function renderSegments(segments, preferredIndex = null) {
  tbody.innerHTML = '';
  activeSegmentIndex = -1;
  cachedSegments = (segments || []).map((segment) => ({
    id: segment.id,
    start: Number(segment.start),
    end: Number(segment.end),
    speaker: segment.speaker,
    text: segment.text,
    items: segment.items || null,
    display_end: segment.display_end == null ? null : Number(segment.display_end),
    confidence: segment.confidence == null ? null : Number(segment.confidence),
    quality_flags: segment.quality_flags || null,
    quality_reasons: segment.quality_reasons || null
  }));
  renderSpeakerMap(segments);
  renderTimeline(segments);
  refreshSpeakerDots();
  if (preferredIndex != null && cachedSegments[preferredIndex] && tableWrap) {
    tableWrap.scrollTop = Math.max(0, preferredIndex * TABLE_ROW_HEIGHT - tableWrap.clientHeight * 0.35);
  }
  renderVisibleSegmentRows();
  if (preferredIndex != null && segments[preferredIndex]) {
    setActiveSegment(preferredIndex, false);
    updateSubtitlePreview(segments);
  } else {
    syncActiveSegment();
  }
}

function createSegmentRow(segment, index) {
  const tr = document.createElement('tr');
  tr.dataset.id = segment.id;
  tr.dataset.index = String(index);
  const speakerList = (cachedSegments || []).map((item) => item.speaker);
  const dotColor = document.querySelector('#speakerColors').value === 'true'
    && new Set(speakerList.filter(Boolean)).size > 1 && segment.speaker
    ? speakerColorOf(segment.speaker, speakerList)
    : 'transparent';
  tr.innerHTML = `
    <td><input class="start" type="number" min="0" step="0.01" value="${segment.start}"></td>
    <td><input class="end" type="number" min="0" step="0.01" value="${segment.end}"></td>
    <td class="speaker-cell"><span class="speaker-dot" style="background:${dotColor}"></span><input class="speaker" type="text" value="${escapeHtml(segment.speaker)}"></td>
    <td><textarea class="text" rows="1" title="Ctrl+Enter：在光标处拆分（按词级时间戳对齐到词边界）">${escapeHtml(segment.text)}</textarea>${segment.confidence != null ? `<div class="quality-score ${segment.quality_flags?.length ? "warning" : ""}" title="${escapeHtml((segment.quality_reasons || []).join("；"))}">置信度 ${Math.round(segment.confidence * 100)}%${segment.quality_flags?.length ? " · ⚠ 需复核" : ""}</div>` : ""}</td>
    <td>
      <div class="segment-actions">
        <button class="segment-action add-row-above" type="button" title="在上方添加字幕">↑+</button>
        <button class="segment-action add-row-below" type="button" title="在下方添加字幕">↓+</button>
        <button class="segment-action merge-row-below" type="button" title="与下方字幕合并"${index === cachedSegments.length - 1 ? ' disabled' : ''}>⇊</button>
        <button class="segment-action delete-row" type="button" title="删除这条字幕">−</button>
      </div>
    </td>
  `;
  tr.addEventListener('click', (event) => {
    if (event.target.closest('button')) return;
    const rowIndex = Number(tr.dataset.index);
    seekPreviewToSegment(rowIndex);
    const isFieldClick = !!event.target.closest('input, textarea');
    setActiveSegment(rowIndex, !isFieldClick, { align: 'center' });
    updateTimelinePlayhead();
    updateSubtitlePreview();
  });
  resizeSegmentRow(tr, false);
  return tr;
}

function renderVisibleSegmentRows() {
  if (!cachedSegments) return;
  const total = cachedSegments.length;
  const container = tableWrap || tbody.closest('.table-wrap');
  const scrollTop = container ? container.scrollTop : 0;
  const viewportHeight = container ? container.clientHeight : 600;
  const start = Math.max(0, Math.floor(scrollTop / TABLE_ROW_HEIGHT) - TABLE_BUFFER_ROWS);
  const visibleCount = Math.ceil(viewportHeight / TABLE_ROW_HEIGHT) + TABLE_BUFFER_ROWS * 2;
  const end = Math.min(total, start + visibleCount);
  const fragment = document.createDocumentFragment();
  if (start > 0) fragment.appendChild(createVirtualSpacer(start * TABLE_ROW_HEIGHT));
  for (let index = start; index < end; index++) {
    fragment.appendChild(createSegmentRow(cachedSegments[index], index));
  }
  if (end < total) fragment.appendChild(createVirtualSpacer((total - end) * TABLE_ROW_HEIGHT));
  tbody.replaceChildren(fragment);
  updateRenderedActiveRows();
}

function createVirtualSpacer(height) {
  const tr = document.createElement('tr');
  tr.className = 'virtual-spacer';
  tr.innerHTML = `<td colspan="5" style="height:${Math.max(0, Math.round(height))}px"></td>`;
  return tr;
}

function scheduleVisibleSegmentRowsRender() {
  if (tableRenderFrame) return;
  tableRenderFrame = requestAnimationFrame(() => {
    tableRenderFrame = 0;
    renderVisibleSegmentRows();
  });
}

function updateRenderedActiveRows() {
  for (const tr of tbody.querySelectorAll('tr[data-index]')) {
    const active = Number(tr.dataset.index) === activeSegmentIndex;
    tr.classList.toggle('active', active);
    resizeSegmentRow(tr, active);
  }
}

function renderTimeline(segments) {
  const duration = timelineDuration(segments);
  const scrollWidth = timelineScroll.clientWidth || 1;
  const pixelsPerSecond = timelinePixelsPerSecond(duration, scrollWidth);
  currentPixelsPerSecond = pixelsPerSecond;
  const trackWidth = Math.max(scrollWidth, Math.ceil(duration * pixelsPerSecond));
  const layout = timelineLaneLayout(segments);
  cachedTimelineSegments = segments;
  cachedTimelineLayout = layout;
  const laneHeight = 44;
  const laneTop = 42;
  const laneCount = Math.max(1, layout.count);
  const laneAreaHeight = laneTop + laneCount * laneHeight + 14;
  timelineTrack.style.width = trackWidth + 'px';
  timelineTrack.style.height = Math.max(timelineScroll.clientHeight, 32 + laneAreaHeight) + 'px';
  timelineLane.style.height = laneAreaHeight + 'px';
  timelineMeta.textContent = segments.length + ' 段' + (duration ? ' · ' + formatTimelineTime(duration) : '') + (laneCount > 1 ? ' · ' + laneCount + ' 层' : '');
  timelineRuler.innerHTML = '';
  timelineLane.innerHTML = '';
  renderTimelineTicks(duration, pixelsPerSecond);
  renderVisibleTimelineSegments();
  timelineTrack.appendChild(timelineClipRange);
  timelineTrack.appendChild(timelinePlayhead);
  updateTimelineClipRange();
  updateTimelinePlayhead(segments);
}

function renderVisibleTimelineSegments() {
  const segments = cachedTimelineSegments || [];
  const layout = cachedTimelineLayout || { lanes: new Map(), count: 1 };
  const pixelsPerSecond = currentPixelsPerSecond || 1;
  const laneHeight = 44;
  const laneTop = 42;
  const leftTime = Math.max(0, (timelineScroll.scrollLeft - TIMELINE_BUFFER_PX) / pixelsPerSecond);
  const rightTime = (timelineScroll.scrollLeft + timelineScroll.clientWidth + TIMELINE_BUFFER_PX) / pixelsPerSecond;
  timelineLane.innerHTML = '';
  for (const [index, segment] of segments.entries()) {
    const start = Math.max(0, Number(segment.start) || 0);
    const end = Math.max(start + 0.01, Number(segment.end) || start + 0.01);
    if (end < leftTime || start > rightTime) continue;
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'timeline-segment' + (index === activeSegmentIndex ? ' active' : '');
    item.dataset.index = String(index);
    item.style.left = Math.max(0, start * pixelsPerSecond) + 'px';
    item.style.top = (laneTop + (layout.lanes.get(index) || 0) * laneHeight) + 'px';
    item.style.width = Math.max(8, (end - start) * pixelsPerSecond) + 'px';
    item.title = `${formatTimelineTime(start)} - ${formatTimelineTime(end)} ${segment.text || ''}`;
    item.innerHTML = `
      <span class="timeline-segment-speaker">${escapeHtml(segment.speaker || 'S--')}</span>
      <span class="timeline-segment-text">${escapeHtml(segment.text || '')}</span>
    `;
    item.addEventListener('pointerdown', (event) => onSegmentPointerDown(event, index, item));
    item.addEventListener('click', (event) => {
      event.preventDefault();
      if (segmentDragState && segmentDragState.moved) {
        return;
      }
      preview.currentTime = start;
      setActiveSegment(index, true, { align: 'center' });
      updateSubtitlePreview();
      updateTimelinePlayhead();
    });
    timelineLane.appendChild(item);
  }
}

function scheduleVisibleTimelineRender() {
  if (timelineRenderFrame) return;
  timelineRenderFrame = requestAnimationFrame(() => {
    timelineRenderFrame = 0;
    renderVisibleTimelineSegments();
  });
}

function updateTimelineClipRange() {
  if (!timelineClipRange || !currentJob) return;
  const start = Math.max(0, Number(clipStartInput.value || 0));
  const end = Math.max(start, Number(clipEndInput.value || 0));
  if (!(end > start)) {
    timelineClipRange.classList.remove('visible');
    return;
  }
  timelineClipRange.style.left = Math.round(start * currentPixelsPerSecond) + 'px';
  timelineClipRange.style.width = Math.max(2, Math.round((end - start) * currentPixelsPerSecond)) + 'px';
  const label = timelineClipRange.querySelector('.timeline-clip-label');
  if (label) label.textContent = `${formatTimelineTime(start)} - ${formatTimelineTime(end)}`;
  timelineClipRange.classList.add('visible');
}

function onClipRangePointerDown(event) {
  if (event.button !== 0 || !currentJob) return;
  const clip = activeClip();
  if (!clip) {
    clipStatusEl.textContent = '先从已选切片里选中一个片段。';
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  const rect = timelineClipRange.getBoundingClientRect();
  const offsetX = event.clientX - rect.left;
  const edgeZone = Math.min(18, Math.max(8, rect.width / 3));
  let mode = event.target.dataset.clipDrag || 'move';
  if (mode !== 'start' && mode !== 'end') {
    if (offsetX <= edgeZone) mode = 'start';
    else if (offsetX >= rect.width - edgeZone) mode = 'end';
    else mode = 'move';
  }
  const segments = collectSegments();
  const duration = timelineDuration(segments);
  clipDragState = {
    clipId: clip.id,
    mode,
    pointerId: event.pointerId,
    startX: timelineContentXFromPointer(event),
    origStart: Math.max(0, Number(clip.start) || 0),
    origEnd: Math.max(Number(clip.start) || 0.25, Number(clip.end) || 0.25),
    duration,
    pps: currentPixelsPerSecond || timelinePixelsPerSecond(duration, timelineScroll.clientWidth || 1),
    segments,
    moved: false,
    newStart: null,
    newEnd: null
  };
  timelineClipRange.classList.add('dragging');
  try { timelineClipRange.setPointerCapture(event.pointerId); } catch (err) {}
  const moveHandler = (ev) => onClipRangePointerMove(ev, clipDragState);
  const upHandler = (ev) => {
    onClipRangePointerUp(ev, clipDragState);
    window.removeEventListener('pointermove', moveHandler);
    window.removeEventListener('pointerup', upHandler);
    window.removeEventListener('pointercancel', upHandler);
  };
  window.addEventListener('pointermove', moveHandler);
  window.addEventListener('pointerup', upHandler);
  window.addEventListener('pointercancel', upHandler);
}

function onClipRangePointerMove(event, state, options = {}) {
  if (!state) return;
  if (!options.fromAutoScroll) updateTimelineEdgeAutoScroll(event, 'clip');
  const dx = timelineContentXFromPointer(event) - state.startX;
  if (!state.moved && Math.abs(dx) < SEGMENT_DRAG_THRESHOLD) return;
  state.moved = true;
  const deltaSec = dx / Math.max(1, state.pps);
  const minDuration = 0.25;
  let newStart = state.origStart;
  let newEnd = state.origEnd;
  if (state.mode === 'move') {
    const clipDuration = state.origEnd - state.origStart;
    newStart = state.origStart + deltaSec;
    newEnd = state.origEnd + deltaSec;
    if (newStart < 0) { newStart = 0; newEnd = clipDuration; }
    if (state.duration > 0 && newEnd > state.duration) {
      newEnd = state.duration;
      newStart = Math.max(0, newEnd - clipDuration);
    }
  } else if (state.mode === 'start') {
    newStart = Math.max(0, Math.min(state.origEnd - minDuration, state.origStart + deltaSec));
  } else {
    newEnd = Math.max(state.origStart + minDuration, state.origEnd + deltaSec);
    if (state.duration > 0) newEnd = Math.min(state.duration, newEnd);
  }
  const snap = computeClipSnap(state, newStart, newEnd);
  if (snap) {
    if (snap.edge === 'start') {
      const shift = snap.time - newStart;
      newStart = snap.time;
      if (state.mode === 'move') newEnd += shift;
    } else {
      const shift = snap.time - newEnd;
      newEnd = snap.time;
      if (state.mode === 'move') newStart += shift;
    }
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (state.duration > 0 && newEnd > state.duration) {
      newStart -= (newEnd - state.duration);
      newEnd = state.duration;
    }
    if (newEnd - newStart < minDuration) {
      if (snap.edge === 'start') newStart = newEnd - minDuration;
      else newEnd = newStart + minDuration;
    }
  }
  if (snap) showTimelineGuide(snap.time, snap.label);
  else hideTimelineGuide();
  state.newStart = Math.max(0, newStart);
  state.newEnd = Math.max(state.newStart + minDuration, newEnd);
  updateClipFromValues(state.clipId, { start: state.newStart, end: state.newEnd, seek: state.mode !== 'end' });
  clipStatusEl.textContent = `正在调整切片：${formatTimelineTime(state.newStart)} - ${formatTimelineTime(state.newEnd)}`;
}

function computeClipSnap(state, newStart, newEnd) {
  return computeSegmentSnap(
    {
      index: -1,
      mode: state.mode,
      pps: state.pps,
      segments: state.segments || [],
    },
    newStart,
    newEnd
  );
}

function onClipRangePointerUp(event, state) {
  if (!state) return;
  timelineClipRange.classList.remove('dragging');
  stopTimelineEdgeAutoScroll();
  hideTimelineGuide();
  try { timelineClipRange.releasePointerCapture(event.pointerId); } catch (err) {}
  if (state.moved && state.newStart != null && state.newEnd != null) {
    updateClipFromValues(state.clipId, { start: state.newStart, end: state.newEnd, seek: state.mode !== 'end' });
    clipStatusEl.textContent = `已调整切片：${formatTimelineTime(state.newStart)} - ${formatTimelineTime(state.newEnd)}。`;
  }
  clipDragState = null;
}

function onSegmentPointerDown(event, index, segment) {
  if (event.button !== 0) return;
  const rect = segment.getBoundingClientRect();
  const offsetX = event.clientX - rect.left;
  const edgeZone = Math.min(SEGMENT_EDGE_PX, Math.max(4, rect.width / 2));
  let mode = null;
  if (offsetX <= edgeZone) mode = 'start';
  else if (offsetX >= rect.width - edgeZone) mode = 'end';
  if (!mode) return;
  event.preventDefault();
  const segments = collectSegments();
  const seg = segments[index];
  if (!seg) return;
  const duration = timelineDuration(segments);
  const pps = currentPixelsPerSecond || timelinePixelsPerSecond(duration, timelineScroll.clientWidth || 1);
  const startX = timelineContentXFromPointer(event);
  segmentDragState = {
    index,
    mode,
    segment,
    pointerId: event.pointerId,
    startX,
    origStart: Math.max(0, Number(seg.start) || 0),
    origEnd: Math.max(Number(seg.start) || 0, Number(seg.end) || 0),
    duration,
    pps,
    segments,
    moved: false,
    newStart: null,
    newEnd: null
  };
  const moveHandler = (ev) => onSegmentPointerMove(ev, segmentDragState);
  const upHandler = (ev) => {
    onSegmentPointerUp(ev, segmentDragState);
    detach();
  };
  const cancelHandler = (ev) => {
    // 系统打断指针（Alt+Tab 切窗/触摸手势）：按拖拽结束处理，
    // 否则监听器与 segmentDragState 永久残留
    onSegmentPointerUp(ev, segmentDragState);
    detach();
  };
  const detach = () => {
    window.removeEventListener('pointermove', moveHandler);
    window.removeEventListener('pointerup', upHandler);
    window.removeEventListener('pointercancel', cancelHandler);
  };
  window.addEventListener('pointermove', moveHandler);
  window.addEventListener('pointerup', upHandler);
  window.addEventListener('pointercancel', cancelHandler);
}

function onSegmentPointerMove(event, state, options = {}) {
  if (!state) return;
  if (!options.fromAutoScroll) updateTimelineEdgeAutoScroll(event, 'segment');
  const dx = timelineContentXFromPointer(event) - state.startX;
  if (!state.moved && Math.abs(dx) < SEGMENT_DRAG_THRESHOLD) return;
  if (!state.moved) {
    state.moved = true;
    state.segment.classList.add('dragging');
  }
  const deltaSec = dx / (Math.max(1, state.pps) * SEGMENT_DRAG_SENSITIVITY);
  let newStart = state.origStart;
  let newEnd = state.origEnd;
  if (state.mode === 'move') {
    newStart = state.origStart + deltaSec;
    newEnd = state.origEnd + deltaSec;
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (state.duration > 0 && newEnd > state.duration) {
      newStart -= (newEnd - state.duration);
      newEnd = state.duration;
    }
  } else if (state.mode === 'start') {
    newStart = Math.max(0, Math.min(state.origEnd - 0.1, state.origStart + deltaSec));
  } else {
    newEnd = Math.max(state.origStart + 0.1, state.origEnd + deltaSec);
    if (state.duration > 0) newEnd = Math.min(state.duration, newEnd);
  }
  const snap = computeSegmentSnap(state, newStart, newEnd);
  if (snap) {
    if (snap.edge === 'start') {
      const shift = snap.time - newStart;
      newStart = snap.time;
      if (state.mode === 'move') newEnd += shift;
    } else {
      const shift = snap.time - newEnd;
      newEnd = snap.time;
      if (state.mode === 'move') newStart += shift;
    }
    if (newStart < 0) { newEnd -= newStart; newStart = 0; }
    if (state.duration > 0 && newEnd > state.duration) {
      newStart -= (newEnd - state.duration);
      newEnd = state.duration;
    }
  }
  if (snap) showTimelineGuide(snap.time, snap.label);
  else hideTimelineGuide();
  state.newStart = newStart;
  state.newEnd = newEnd;
  state.segment.style.left = Math.max(0, newStart * state.pps) + 'px';
  state.segment.style.width = Math.max(8, (newEnd - newStart) * state.pps) + 'px';
}

function computeSegmentSnap(state, newStart, newEnd) {
  const pps = state.pps || 1;
  const threshold = Math.max(0.05, SNAP_PX / pps);
  const candidates = [
    { time: 0, label: '起点' },
    { time: Number(preview.currentTime || 0), label: '播放头' }
  ];
  const segments = state.segments;
  const addCandidateEdges = (i) => {
    if (i === state.index || !segments[i]) return;
    candidates.push(
      { time: Number(segments[i].start) || 0, label: '头对齐' },
      { time: Number(segments[i].end) || 0, label: '尾对齐' }
    );
  };
  if (segments.length > 800) {
    const center = findSegmentIndexAtTime(segments, (newStart + newEnd) / 2);
    for (let i = Math.max(0, center - 8); i < Math.min(segments.length, center + 9); i++) addCandidateEdges(i);
  } else {
    for (let i = 0; i < segments.length; i++) addCandidateEdges(i);
  }
  const edges = state.mode === 'end'
    ? [{ edge: 'end', time: newEnd }]
    : state.mode === 'start'
      ? [{ edge: 'start', time: newStart }]
      : [{ edge: 'start', time: newStart }, { edge: 'end', time: newEnd }];
  let best = null;
  for (const edge of edges) {
    for (const cand of candidates) {
      if (!Number.isFinite(cand.time)) continue;
      const d = Math.abs(edge.time - cand.time);
      if (d <= threshold && (!best || d < best.dist)) {
        best = { edge: edge.edge, time: cand.time, label: cand.label, dist: d };
      }
    }
  }
  return best;
}

function onSegmentPointerUp(event, state) {
  if (!state) return;
  state.segment.classList.remove('dragging');
  stopTimelineEdgeAutoScroll();
  hideTimelineGuide();
  try { state.segment.releasePointerCapture(event.pointerId); } catch (err) {}
  if (state.moved && state.newStart != null && state.newEnd != null) {
    if (!editorDirty) pushUndoSnapshot();
    const tr = tbody.querySelector('tr[data-index="' + state.index + '"]');
    if (tr) {
      tr.querySelector('.start').value = roundTime(state.newStart);
      tr.querySelector('.end').value = roundTime(state.newEnd);
      updateCachedSegmentFromRow(tr);
    } else if (cachedSegments && cachedSegments[state.index]) {
      cachedSegments[state.index] = {
        ...cachedSegments[state.index],
        start: roundTime(state.newStart),
        end: roundTime(state.newEnd),
      };
    }
    markEditorDirty();
    renderTimeline(collectSegments());
    updateSubtitlePreview();
    updateTimelinePlayhead();
  }
  const moved = state.moved;
  segmentDragState = moved ? { moved: true } : null;
  if (moved) {
    const st = segmentDragState;
    setTimeout(() => { if (segmentDragState === st) segmentDragState = null; }, 60);
  }
}

function showTimelineGuide(time, label) {
  const left = timelineTimeToX(time);
  timelineGuide.style.left = left + 'px';
  const minLabelOffset = 28;
  const maxLabelOffset = Math.max(minLabelOffset, timelineScroll.clientWidth - 44);
  const viewportLeft = left - timelineScroll.scrollLeft;
  const clampedOffset = Math.max(minLabelOffset, Math.min(maxLabelOffset, viewportLeft));
  timelineGuide.style.setProperty('--guide-label-offset', (clampedOffset - viewportLeft) + 'px');
  timelineGuide.dataset.label = label || '对齐';
  timelineGuide.classList.add('visible');
  timelineGuide.classList.add('snapped');
}

function hideTimelineGuide() {
  timelineGuide.classList.remove('visible');
  timelineGuide.classList.remove('snapped');
}

function timelineLaneLayout(segments) {
  const items = segments
    .map((segment, index) => ({
      index,
      start: Math.max(0, Number(segment.start) || 0),
      end: Math.max(Number(segment.start) || 0, Number(segment.end) || Number(segment.start) || 0)
    }))
    .sort((a, b) => a.start - b.start || a.end - b.end || a.index - b.index);
  const laneEnds = [];
  const lanes = new Map();
  for (const item of items) {
    let lane = laneEnds.findIndex((end) => end <= item.start + 0.001);
    if (lane < 0) {
      lane = laneEnds.length;
      laneEnds.push(0);
    }
    laneEnds[lane] = Math.max(item.end, item.start + 0.01);
    lanes.set(item.index, lane);
  }
  return { lanes, count: laneEnds.length };
}

function timelinePixelsPerSecond(duration, scrollWidth) {
  if (!duration || duration <= 0) return 12;
  const base = Math.max(1800, scrollWidth);
  if (duration <= 180) {
    return Math.max(48, Math.min(72, Math.max(base * 2, 4800) / duration));
  }
  return Math.max(8, Math.min(32, base / duration));
}

function renderTimelineTicks(duration, pixelsPerSecond) {
  const interval = timelineTickInterval(pixelsPerSecond);
  const end = Math.max(interval, Math.ceil((duration || interval) / interval) * interval);
  for (let time = 0; time <= end; time += interval) {
    const tick = document.createElement('div');
    tick.className = 'timeline-tick major';
    tick.style.left = Math.round(time * pixelsPerSecond) + 'px';
    const label = document.createElement('span');
    label.textContent = formatTimelineTime(time);
    tick.appendChild(label);
    timelineRuler.appendChild(tick);
    const half = time + interval / 2;
    if (half < end) {
      const minor = document.createElement('div');
      minor.className = 'timeline-tick';
      minor.style.left = Math.round(half * pixelsPerSecond) + 'px';
      timelineRuler.appendChild(minor);
    }
  }
}

function timelineTickInterval(pixelsPerSecond) {
  if (pixelsPerSecond >= 24) return 5;
  if (pixelsPerSecond >= 14) return 10;
  return 15;
}

function timelineDuration(segments) {
  const mediaDuration = Number(preview.duration || 0);
  const segmentDuration = Math.max(0, ...segments.map((segment) => Number(segment.end) || 0));
  return Math.max(mediaDuration, segmentDuration);
}

function formatTimelineTime(seconds) {
  seconds = Math.max(0, Number(seconds) || 0);
  const total = Math.floor(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function resizeSegmentTextarea(textarea, expanded) {
  if (!textarea) return;
  const maxHeight = expanded ? 112 : 48;
  textarea.style.height = 'auto';
  const naturalHeight = textarea.scrollHeight;
  const nextHeight = Math.max(30, Math.min(naturalHeight, maxHeight));
  textarea.style.height = nextHeight + 'px';
  textarea.style.overflowY = naturalHeight > maxHeight ? 'auto' : 'hidden';
}

function resizeSegmentRow(tr, expanded) {
  resizeSegmentTextarea(tr && tr.querySelector('textarea.text'), expanded);
}

function collectSegments() {
  if (cachedSegments) return cachedSegments.map((segment) => ({ ...segment }));
  cachedSegments = [...tbody.querySelectorAll('tr')].map((tr, index) => ({
    id: tr.dataset.id || `seg_${String(index + 1).padStart(4, '0')}`,
    start: Number(tr.querySelector('.start').value),
    end: Number(tr.querySelector('.end').value),
    speaker: tr.querySelector('.speaker').value,
    text: tr.querySelector('.text').value
  }));
  return cachedSegments.map((segment) => ({ ...segment }));
}

function updateCachedSegmentFromRow(tr) {
  if (!cachedSegments || !tr) return;
  const index = Number(tr.dataset.index);
  if (!Number.isInteger(index) || !cachedSegments[index]) return;
  cachedSegments[index] = {
    id: tr.dataset.id || cachedSegments[index].id || `seg_${String(index + 1).padStart(4, '0')}`,
    start: Number(tr.querySelector('.start').value),
    end: Number(tr.querySelector('.end').value),
    speaker: tr.querySelector('.speaker').value,
    text: tr.querySelector('.text').value
  };
}

function nudgeActiveSegment(kind, delta) {
  if (!currentJob || !cachedSegments || activeSegmentIndex < 0) return;
  const segment = cachedSegments[activeSegmentIndex];
  if (!segment) return;
  const minDuration = 0.1;
  const mediaEnd = Number(preview.duration || 0);
  let start = Math.max(0, Number(segment.start) || 0);
  let end = Math.max(start + minDuration, Number(segment.end) || start + minDuration);
  if (kind === 'shift') {
    const duration = end - start;
    start += delta;
    end += delta;
    if (start < 0) {
      start = 0;
      end = duration;
    }
    if (mediaEnd > 0 && end > mediaEnd) {
      end = mediaEnd;
      start = Math.max(0, end - duration);
    }
  } else if (kind === 'start') {
    start = Math.max(0, Math.min(end - minDuration, start + delta));
  } else if (kind === 'end') {
    end = Math.max(start + minDuration, end + delta);
    if (mediaEnd > 0) end = Math.min(mediaEnd, end);
  }
  applySegmentTiming(activeSegmentIndex, roundTime(start), roundTime(end), { seek: true });
}

function applySegmentTiming(index, start, end, options = {}) {
  if (!cachedSegments || !cachedSegments[index]) return;
  // 微调/拖拽会连续触发：只在每轮修改的第一次推基线快照，避免快照刷屏
  if (!editorDirty) pushUndoSnapshot();
  cachedSegments[index] = {
    ...cachedSegments[index],
    start,
    end,
  };
  const tr = tbody.querySelector('tr[data-index="' + index + '"]');
  if (tr) {
    tr.querySelector('.start').value = start;
    tr.querySelector('.end').value = end;
  }
  markEditorDirty();
  renderTimeline(collectSegments());
  setActiveSegment(index, true, { align: 'center' });
  if (options.seek) seekPreviewToSegment(index);
  updateTimelinePlayhead();
  updateSubtitlePreview();
}

function seekPreviewToSegment(index, options = {}) {
  const segment = cachedSegments && cachedSegments[index];
  if (!segment) return false;
  const start = Number(segment.start);
  if (!Number.isFinite(start)) return false;
  if (options.pause !== false) preview.pause();
  lastSyncedTime = -1;
  preview.currentTime = Math.max(0, start);
  timelineFollowHoldUntil = 0;
  scrollTimelineTimeIntoView(start, { align: options.align || 'center' });
  syncMaskPreviewTime();
  return true;
}

function scrollTimelineTimeIntoView(time, options = {}) {
  if (!timelineScroll || !Number.isFinite(Number(time))) return;
  const duration = timelineDuration(cachedTimelineSegments.length ? cachedTimelineSegments : collectSegments());
  if (duration <= 0) return;
  const containerWidth = timelineScroll.clientWidth || 0;
  if (containerWidth <= 0) return;
  const trackWidth = Number.parseFloat(timelineTrack.style.width) || timelineTrack.scrollWidth || containerWidth;
  const maxScrollLeft = Math.max(0, trackWidth - containerWidth);
  const x = timelineTimeToX(time);
  let nextScrollLeft = timelineScroll.scrollLeft;
  if (options.align === 'center') {
    nextScrollLeft = Math.max(0, Math.min(maxScrollLeft, x - containerWidth * 0.5));
  } else if (x < timelineScroll.scrollLeft + 24) {
    nextScrollLeft = Math.max(0, x - 24);
  } else if (x > timelineScroll.scrollLeft + containerWidth - 24) {
    nextScrollLeft = Math.max(0, Math.min(maxScrollLeft, x - containerWidth + 24));
  } else {
    return;
  }
  if (Math.abs(nextScrollLeft - timelineScroll.scrollLeft) > 0.5) {
    timelineScroll.scrollLeft = nextScrollLeft;
    renderVisibleTimelineSegments();
  }
}

function addSegmentAtPlayhead() {
  if (!currentJob) return;
  pushUndoSnapshot();
  const segments = collectSegments();
  const start = roundTime(Math.max(0, Number(preview.currentTime || 0)));
  const next = segments.find((segment) => Number(segment.start) > start);
  const mediaEnd = Number(preview.duration || 0);
  const defaultEnd = mediaEnd > 0 ? Math.min(mediaEnd, start + 2.5) : start + 2.5;
  const end = roundTime(Math.max(start + 0.25, next ? Math.min(Number(next.start), defaultEnd) : defaultEnd));
  const currentSpeaker = segments[activeSegmentIndex] && segments[activeSegmentIndex].speaker;
  const segment = {
    id: 'seg_' + Date.now().toString(36),
    start,
    end,
    speaker: currentSpeaker || 'S01',
    text: ''
  };
  segments.push(segment);
  segments.sort((a, b) => Number(a.start) - Number(b.start));
  const index = segments.findIndex((item) => item.id === segment.id);
  preview.currentTime = start;
  renderSegments(segments, index);
  markEditorDirty();
  focusSegmentText(index);
}

function addSegmentAroundIndex(index, placement) {
  if (!currentJob) return;
  const segments = collectSegments();
  const source = segments[index];
  if (!source) {
    addSegmentAtPlayhead();
    return;
  }
  const isAbove = placement === 'above';
  pushUndoSnapshot();
  const previous = segments[index - 1];
  const next = segments[index + 1];
  const anchorStart = Math.max(0, Number(source.start) || 0);
  const anchorEnd = Math.max(anchorStart, Number(source.end) || anchorStart);
  const mediaEnd = Number(preview.duration || 0);
  const segment = createBlankAdjacentSegment({
    source,
    previous,
    next,
    anchorStart,
    anchorEnd,
    mediaEnd,
    isAbove
  });
  const insertIndex = isAbove ? index : index + 1;
  segments.splice(insertIndex, 0, segment);
  preview.currentTime = segment.start;
  renderSegments(segments, insertIndex);
  markEditorDirty();
  focusSegmentText(insertIndex);
}

function createBlankAdjacentSegment({ source, previous, next, anchorStart, anchorEnd, mediaEnd, isAbove }) {
  let start;
  let end;
  if (isAbove) {
    end = anchorStart;
    const floor = previous ? Math.max(0, Number(previous.end) || 0) : 0;
    start = Math.max(floor, end - 2.5);
    if (end - start < 0.25) {
      start = Math.max(0, end - 0.25);
      if (end <= start) end = start + 0.25;
    }
  } else {
    start = anchorEnd;
    const ceiling = next ? Number(next.start) : (mediaEnd > 0 ? mediaEnd : start + 2.5);
    const defaultEnd = mediaEnd > 0 ? Math.min(mediaEnd, start + 2.5) : start + 2.5;
    end = Math.max(start + 0.25, Math.min(Number.isFinite(ceiling) ? ceiling : defaultEnd, defaultEnd));
  }
  if (mediaEnd > 0) {
    start = Math.min(start, Math.max(0, mediaEnd - 0.25));
    end = Math.min(Math.max(start + 0.25, end), mediaEnd);
  }
  start = roundTime(start);
  end = roundTime(Math.max(start + 0.25, end));
  if (mediaEnd > 0) end = roundTime(Math.min(end, mediaEnd));
  const segment = {
    id: 'seg_' + Date.now().toString(36),
    start,
    end,
    speaker: source.speaker || 'S01',
    text: ''
  };
  return segment;
}

function deleteActiveSegment() {
  if (!currentJob || activeSegmentIndex < 0) return;
  deleteSegmentAtIndex(activeSegmentIndex);
}

function deleteSegmentAtIndex(index) {
  if (!currentJob || index < 0) return;
  pushUndoSnapshot();
  const segments = collectSegments();
  if (!segments[index]) return;
  segments.splice(index, 1);
  const nextIndex = Math.min(index, segments.length - 1);
  renderSegments(segments, nextIndex >= 0 ? nextIndex : null);
  markEditorDirty();
}

async function splitSegmentAtCursor(index, textarea) {
  if (!currentJob) return;
  const segment = (cachedSegments || [])[index];
  if (!segment) return;
  // 拆分是服务端操作(需要词级 items),先把未保存的编辑落盘
  if (editorDirty && !(await saveSegments())) return;
  pushUndoSnapshot();
  const fresh = (cachedSegments || [])[index] || segment;
  const cursor = textarea.selectionStart != null ? Number(textarea.selectionStart) : 0;
  const text = textarea.value != null ? textarea.value : fresh.text;

  // 光标位置 -> 词索引 -> 精确切点时间
  let time = null;
  const items = fresh.items;
  if (items && items.length) {
    let pos = 0;
    for (let k = 0; k < items.length; k++) {
      const word = String(items[k].text || '').trim();
      if (!word) continue;
      const idx = text.indexOf(word, pos);
      if (idx === -1) continue;
      if (cursor <= idx) { time = Number(items[k].start); break; }
      pos = idx + word.length;
    }
  }
  if (time == null) {
    // 无词级数据(或光标在末尾): 按字符占比估时间,后端吸附到最近词边界
    const ratio = text.length ? Math.min(1, Math.max(0, cursor / text.length)) : 0.5;
    time = Number(fresh.start) + (Number(fresh.end) - Number(fresh.start)) * ratio;
  }
  if (time <= Number(fresh.start) + 0.05 || time >= Number(fresh.end) - 0.05) {
    setTaskNotice('光标太靠边，无法在此拆分', 'error');
    return;
  }
  setTaskNotice('正在拆分...', '');
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments/${encodeURIComponent(fresh.id)}/split`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ time })
    });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '拆分失败', 'error');
      return;
    }
    setTaskNotice('', '');
    renderSegments(data.segments, index + 1);
    setEditorDirty(false);
    updateSubtitlePreview(data.segments);
    focusSegmentText(index + 1);
    if (data.needs_retranslate) setTaskNotice('已拆分：源稿已同步，译文段落将显示原文，请重新翻译', 'warn');
  } catch (err) {
    setTaskNotice('拆分失败：' + err.message, 'error');
  }
}

async function mergeSegmentWithNext(index) {
  if (!currentJob) return;
  const segments = collectSegments();
  const current = segments[index];
  const next = segments[index + 1];
  if (!current || !next) return;
  if (editorDirty && !(await saveSegments())) return;
  pushUndoSnapshot();
  setTaskNotice('正在合并...', '');
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/segments/merge`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: [current.id, next.id] })
    });
    const data = await res.json();
    if (!res.ok) {
      setTaskNotice(data.detail || '合并失败', 'error');
      return;
    }
    setTaskNotice('', '');
    renderSegments(data.segments, index);
    setEditorDirty(false);
    updateSubtitlePreview(data.segments);
    if (data.needs_retranslate) setTaskNotice('已合并：源稿已同步，重新翻译时将保留新结构', 'warn');
  } catch (err) {
    setTaskNotice('合并失败：' + err.message, 'error');
  }
}

// -------------------------------------------------------------- 查找与替换

let searchMatches = [];
let searchCursor = -1;
let searchDebounceTimer = 0;

async function runSearch({ keepCursor = false } = {}) {
  if (!currentJob) return;
  const query = searchQueryInput.value.trim();
  if (!query) {
    searchMatches = [];
    searchCursor = -1;
    searchCountEl.textContent = '';
    return;
  }
  // 搜索作用于服务器已保存的内容；有未保存修改先落盘，避免搜到旧稿。
  if (editorDirty && !(await saveSegments())) return;
  try {
    const res = await fetch(
      apiUrl(`api/jobs/${currentJob.id}/search?q=${encodeURIComponent(query)}&mode=${encodeURIComponent(searchModeSelect.value)}`),
      { cache: 'no-store' }
    );
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '搜索失败');
    searchMatches = data.matches || [];
    if (!searchMatches.length) {
      searchCursor = -1;
      searchCountEl.textContent = '无匹配';
      return;
    }
    if (!keepCursor || searchCursor < 0 || searchCursor >= searchMatches.length) searchCursor = 0;
    jumpToSearchMatch(searchCursor);
  } catch (err) {
    searchCountEl.textContent = '搜索失败';
  }
}

function jumpToSearchMatch(index) {
  const match = searchMatches[index];
  if (!match) return;
  searchCursor = index;
  searchCountEl.textContent = `${searchMatches.length} 处 · 第 ${index + 1} 个`;
  setActiveSegment(match.index, true, { align: 'center' });
  seekPreviewToSegment(match.index);
  updateTimelinePlayhead();
}

function stepSearchMatch(delta) {
  if (!searchMatches.length) return;
  const next = (searchCursor + delta + searchMatches.length) % searchMatches.length;
  jumpToSearchMatch(next);
}

searchQueryInput.addEventListener('input', () => {
  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => runSearch(), 400);
});
searchQueryInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    if (searchMatches.length) stepSearchMatch(1);
    else runSearch();
  }
});
searchModeSelect.addEventListener('change', () => runSearch());
searchPrevBtn.addEventListener('click', () => stepSearchMatch(-1));
searchNextBtn.addEventListener('click', () => stepSearchMatch(1));

replaceAllBtn.addEventListener('click', async () => {
  if (!currentJob) return;
  const query = searchQueryInput.value.trim();
  if (!query) return;
  if (editorDirty && !(await saveSegments())) return;
  if (!searchMatches.length) await runSearch();
  if (!searchMatches.length) {
    setTaskNotice('没有可替换的匹配', 'error');
    return;
  }
  const replacement = replaceTextInput.value;
  const label = replacement ? `「${replacement}」` : '空字符串（删除）';
  if (!window.confirm(`确定将 ${searchMatches.length} 处「${query}」替换为 ${label}？`)) return;
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/replace`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, replacement, mode: searchModeSelect.value })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '替换失败');
    pushUndoSnapshot();
    renderSegments(data.segments);
    setEditorDirty(false);
    updateSubtitlePreview(data.segments);
    setTaskNotice(`已替换 ${data.replacements} 处`);
    await runSearch();
  } catch (err) {
    setTaskNotice('替换失败：' + (err.message || err), 'error');
  }
});

function focusSegmentText(index) {
  scrollSegmentIndexIntoView(index);
  const tr = tbody.querySelector(`tr[data-index="${index}"]`);
  const textarea = tr && tr.querySelector('textarea.text');
  if (!textarea) return;
  textarea.focus();
  textarea.select();
}

function roundTime(value) {
  return Math.round(Number(value || 0) * 100) / 100;
}

function timelineTimeToX(time) {
  return Math.max(0, Number(time || 0) * (currentPixelsPerSecond || 1));
}

function timelineXToTime(x, duration) {
  const pps = currentPixelsPerSecond || 1;
  return Math.max(0, Math.min(duration, Number(x || 0) / pps));
}

function timelineViewportXFromPointer(event) {
  const rect = timelineScroll.getBoundingClientRect();
  const width = timelineScroll.clientWidth || rect.width || 1;
  return Math.max(0, Math.min(width, event.clientX - rect.left));
}

function timelineContentXFromPointer(event) {
  const trackWidth = timelineTrack.scrollWidth || timelineTrack.clientWidth || timelineScroll.clientWidth || 1;
  return Math.max(0, Math.min(trackWidth, timelineScroll.scrollLeft + timelineViewportXFromPointer(event)));
}

function updateTimelineEdgeAutoScroll(event, mode) {
  timelineAutoScrollPointer = { clientX: event.clientX, clientY: event.clientY };
  timelineAutoScrollMode = mode;
  if (!timelineAutoScrollFrame && timelineEdgeScrollSpeed(event) !== 0) {
    timelineAutoScrollFrame = requestAnimationFrame(runTimelineEdgeAutoScroll);
  }
}

function timelineEdgeScrollSpeed(event) {
  const rect = timelineScroll.getBoundingClientRect();
  const maxScroll = Math.max(0, (timelineTrack.scrollWidth || timelineTrack.clientWidth || 0) - timelineScroll.clientWidth);
  if (maxScroll <= 0) return 0;
  if (event.clientX < rect.left && timelineScroll.scrollLeft > 0) {
    return -Math.min(14, Math.max(2, (rect.left - event.clientX) * 0.12));
  }
  if (event.clientX > rect.right && timelineScroll.scrollLeft < maxScroll) {
    return Math.min(14, Math.max(2, (event.clientX - rect.right) * 0.12));
  }
  return 0;
}

function runTimelineEdgeAutoScroll() {
  timelineAutoScrollFrame = 0;
  const pointer = timelineAutoScrollPointer;
  if (!pointer || (!timelineDragging && !(segmentDragState && segmentDragState.moved) && !(clipDragState && clipDragState.moved))) return;
  const speed = timelineEdgeScrollSpeed(pointer);
  if (speed === 0) return;
  const maxScroll = Math.max(0, (timelineTrack.scrollWidth || timelineTrack.clientWidth || 0) - timelineScroll.clientWidth);
  const nextScroll = Math.max(0, Math.min(maxScroll, timelineScroll.scrollLeft + speed));
  if (Math.abs(nextScroll - timelineScroll.scrollLeft) > 0.1) {
    timelineScroll.scrollLeft = nextScroll;
    if (timelineAutoScrollMode === 'seek' && timelineDragging) {
      seekTimelineFromPointer(pointer);
    } else if (timelineAutoScrollMode === 'segment' && segmentDragState) {
      onSegmentPointerMove(pointer, segmentDragState, { fromAutoScroll: true });
    } else if (timelineAutoScrollMode === 'clip' && clipDragState) {
      onClipRangePointerMove(pointer, clipDragState, { fromAutoScroll: true });
    }
  }
  timelineAutoScrollFrame = requestAnimationFrame(runTimelineEdgeAutoScroll);
}

function stopTimelineEdgeAutoScroll() {
  timelineAutoScrollPointer = null;
  timelineAutoScrollMode = '';
  if (timelineAutoScrollFrame) {
    cancelAnimationFrame(timelineAutoScrollFrame);
    timelineAutoScrollFrame = 0;
  }
}

function seekTimelineFromPointer(event) {
  const segments = collectSegments();
  const duration = timelineDuration(segments);
  if (!duration) return;
  const x = timelineContentXFromPointer(event);
  const rawTime = roundTime(timelineXToTime(x, duration));
  const time = Math.max(0, Math.min(duration, rawTime));
  preview.currentTime = time;
  hideTimelineGuide();
  syncActiveSegment();
}

function computePlayheadSnap(time, segments) {
  const pps = currentPixelsPerSecond || 1;
  const threshold = Math.max(0.05, SNAP_PX / pps);
  const candidates = [{ time: 0, label: '起点' }];
  const addSegmentSnap = (segment) => {
    if (!segment) return;
    const start = Number(segment.start);
    const end = Number(segment.end);
    if (Number.isFinite(start)) candidates.push({ time: start, label: '头对齐' });
    if (Number.isFinite(end)) candidates.push({ time: end, label: '尾对齐' });
  };
  if (segments.length > 800) {
    const center = findSegmentIndexAtTime(segments, time);
    for (let i = Math.max(0, center - 8); i < Math.min(segments.length, center + 9); i++) addSegmentSnap(segments[i]);
  } else {
    for (const segment of segments) addSegmentSnap(segment);
  }
  let best = null;
  for (const candidate of candidates) {
    const dist = Math.abs(time - candidate.time);
    if (dist <= threshold && (!best || dist < best.dist)) {
      best = { ...candidate, dist };
    }
  }
  return best;
}

function syncMaskPreviewTime() {
  if (!maskPreviewVideo.src) return;
  const drift = Math.abs(Number(maskPreviewVideo.currentTime || 0) - Number(preview.currentTime || 0));
  if (drift > 0.12) {
    try {
      maskPreviewVideo.currentTime = preview.currentTime || 0;
    } catch (err) {}
  }
}

function ensureMaskPreviewSource() {
  if (!preview.currentSrc && !preview.src) return false;
  const source = preview.currentSrc || preview.src;
  if (maskPreviewVideo.src !== source) {
    maskPreviewVideo.src = source;
    maskPreviewVideo.load();
  }
  return true;
}

function unloadMaskPreviewSource() {
  if (!maskPreviewVideo.src) return;
  maskPreviewVideo.pause();
  maskPreviewVideo.removeAttribute('src');
  maskPreviewVideo.load();
}

function syncMaskPreviewPlaybackRate() {
  maskPreviewVideo.playbackRate = preview.playbackRate || 1;
}

function syncMaskPreviewPlayback() {
  if (!maskPreviewVideo.src) return;
  syncMaskPreviewTime();
  syncMaskPreviewPlaybackRate();
  if (preview.paused || preview.ended) {
    maskPreviewVideo.pause();
    return;
  }
  const playPromise = maskPreviewVideo.play();
  if (playPromise && typeof playPromise.catch === 'function') playPromise.catch(() => {});
}

function resetVideoStage() {
  assPlayRes = { x: 1920, y: 1080 };
  videoStage.style.width = '';
  videoStage.style.height = '';
  videoStage.style.aspectRatio = assPlayRes.x + ' / ' + assPlayRes.y;
  maskPreviewVideo.classList.remove('visible');
  maskPreviewVideo.pause();
}

function fitVideoStageToMedia() {
  const videoWidth = Number(preview.videoWidth || 0);
  const videoHeight = Number(preview.videoHeight || 0);
  const shell = videoStage.parentElement;
  if (!shell || videoWidth <= 0 || videoHeight <= 0) {
    resetVideoStage();
    updateSubtitlePreview();
    return;
  }
  assPlayRes = { x: videoWidth, y: videoHeight };
  const maxWidth = shell.clientWidth || videoWidth;
  const maxHeight = Math.max(180, Math.floor(window.innerHeight * 0.48));
  const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight);
  videoStage.style.width = Math.max(1, Math.floor(videoWidth * scale)) + 'px';
  videoStage.style.height = Math.max(1, Math.floor(videoHeight * scale)) + 'px';
  videoStage.style.aspectRatio = videoWidth + ' / ' + videoHeight;
  updateSubtitlePreview();
}

function assScriptScale() {
  const playResY = Number(assPlayRes.y || preview.videoHeight || 0);
  if (playResY <= 0) return 1;
  return (videoStage.clientHeight || playResY) / playResY;
}

function scheduleActiveSegmentSync(force = false) {
  const time = Number(preview.currentTime || 0);
  if (!force && lastSyncedTime >= 0 && Math.abs(time - lastSyncedTime) < 0.08) return;
  if (syncActiveFrame) return;
  syncActiveFrame = requestAnimationFrame(() => {
    syncActiveFrame = 0;
    syncActiveSegment(force);
  });
}

function syncActiveSegment(force = false) {
  syncMaskPreviewTime();
  const time = Number(preview.currentTime || 0);
  if (!force && lastSyncedTime >= 0 && Math.abs(time - lastSyncedTime) < 0.08) return;
  lastSyncedTime = time;
  const segments = collectSegments();
  const previousIndex = activeSegmentIndex;
  const previous = segments[previousIndex];
  const candidateIndex = previous && isSegmentVisibleAtTime(previous, time) ? previousIndex : findSegmentIndexAtTime(segments, time);
  const index = candidateIndex >= 0 ? candidateIndex : previousIndex;
  // 自动跟随滚动的条件：
  //   1. 视频正在播放（暂停时不抢滚动条，让用户自由浏览字幕表）
  //   2. 用户没有在手动滚动字幕表（3 秒内没有 wheel/pointerdown/touchstart）
  //   3. force=true（用户主动 seek）时，也要求"非用户手动滚动"才滚
  // 其余情况：只更新高亮和字幕预览，不滚动表格
  const userScrolling = performance.now() < tableUserScrollUntil;
  const playing = preview && !preview.paused && !preview.ended;
  const shouldScroll = !userScrolling && (force || playing);
  setActiveSegment(index, shouldScroll, { followPlayback: playing });
  updateTimelinePlayhead(segments);
  updateSubtitlePreview(segments);
}

function setActiveSegment(index, shouldScroll, scrollOptions = {}) {
  index = Number(index);
  const validIndex = Number.isInteger(index) && cachedSegments && index >= 0 && index < cachedSegments.length;
  if (!validIndex) {
    if (activeSegmentIndex === -1) return;
    activeSegmentIndex = -1;
    updateRenderedActiveRows();
    for (const item of timelineLane.querySelectorAll('.timeline-segment')) {
      item.classList.remove('active');
    }
    scheduleVisibleTimelineRender();
    return;
  }
  const sameIndex = index === activeSegmentIndex;
  activeSegmentIndex = index;
  if (shouldScroll) scrollSegmentIndexIntoView(index, scrollOptions);
  if (sameIndex && !shouldScroll) return;
  updateRenderedActiveRows();
  for (const item of timelineLane.querySelectorAll('.timeline-segment')) {
    item.classList.toggle('active', Number(item.dataset.index) === index);
  }
}

function updateTimelinePlayhead(segments) {
  segments = segments || collectSegments();
  const duration = timelineDuration(segments);
  const trackWidth = Number.parseFloat(timelineTrack.style.width) || timelineTrack.clientWidth || timelineScroll.clientWidth || 1;
  const time = Math.max(0, Number(preview.currentTime || 0));
  const left = duration > 0 ? Math.min(trackWidth, timelineTimeToX(time)) : 0;
  timelinePlayhead.style.left = left + 'px';
  const now = performance.now();
  if (time > 0 && timelineScroll.clientWidth && now >= timelineFollowHoldUntil && !timelineDragging && !(segmentDragState && segmentDragState.moved) && !(clipDragState && clipDragState.moved)) {
    const visibleLeft = timelineScroll.scrollLeft;
    const visibleRight = visibleLeft + timelineScroll.clientWidth;
    if (left < visibleLeft + 24 || left > visibleRight - 24) {
      timelineScroll.scrollLeft = Math.max(0, left - timelineScroll.clientWidth * 0.45);
    }
  }
}

function scrollSegmentIndexIntoView(index, options = {}) {
  const container = tableWrap || tbody.closest('.table-wrap');
  if (!container) return;
  index = Number(index);
  if (!Number.isInteger(index) || index < 0 || !cachedSegments || index >= cachedSegments.length) return;
  const stickyHeaderHeight = 30;
  // 虚拟列表中的字幕行高度会因换行而不同，不能用 index * 常量估算滚动位置。
  // 已渲染行优先使用真实 DOM 坐标；未渲染行先用估算定位，下一帧再校正。
  const renderedRow = tbody.querySelector(`tr[data-index="${index}"]`);
  const containerRect = container.getBoundingClientRect();
  const rowRect = renderedRow?.getBoundingClientRect();
  const rowTop = rowRect ? container.scrollTop + rowRect.top - containerRect.top : index * TABLE_ROW_HEIGHT;
  const rowBottom = rowRect ? container.scrollTop + rowRect.bottom - containerRect.top : rowTop + TABLE_ROW_HEIGHT;
  const viewTop = container.scrollTop + stickyHeaderHeight;
  const viewBottom = container.scrollTop + container.clientHeight;
  let nextScrollTop = container.scrollTop;
  if (options.align === 'center') {
    nextScrollTop = Math.max(0, rowTop - Math.max(0, (container.clientHeight - TABLE_ROW_HEIGHT) / 2));
  } else if (options.followPlayback) {
    // 播放跟随使用中部安全区：字幕换行时只向前滚一点，避免当前行贴在底部。
    const safeTop = container.scrollTop + Math.max(stickyHeaderHeight + 12, container.clientHeight * 0.28);
    const safeBottom = container.scrollTop + container.clientHeight * 0.72;
    if (rowTop < safeTop) nextScrollTop = Math.max(0, rowTop - container.clientHeight * 0.28);
    else if (rowBottom > safeBottom) nextScrollTop = Math.max(0, rowBottom - container.clientHeight * 0.55);
  } else if (rowTop < viewTop) {
    nextScrollTop = Math.max(0, rowTop - stickyHeaderHeight - 4);
  } else if (rowBottom > viewBottom) {
    nextScrollTop = rowBottom - container.clientHeight + 8;
  }
  if (Math.abs(nextScrollTop - container.scrollTop) > 1) {
    container.scrollTop = nextScrollTop;
    renderVisibleSegmentRows();
    if (!renderedRow) {
      requestAnimationFrame(() => scrollSegmentIndexIntoView(index, options));
    }
  }
}

function updateSubtitlePreview(segments) {
  segments = segments || collectSegments();
  updateSourceMaskPreview();
  const time = Number(preview.currentTime || 0);
  const centerIndex = activeSegmentIndex >= 0 ? activeSegmentIndex : findSegmentIndexAtTime(segments, time);
  const searchStart = Math.max(0, centerIndex - 8);
  const searchEnd = Math.min(segments.length, Math.max(centerIndex + 9, 0));
  const visibleSegments = segments
    .slice(searchStart, searchEnd)
    .map((segment, offset) => ({ segment, index: searchStart + offset }))
    .filter((item) => isSegmentVisibleAtTime(item.segment, time) && String(item.segment.text || '').trim())
    .sort((a, b) => {
      if (a.index === activeSegmentIndex) return -1;
      if (b.index === activeSegmentIndex) return 1;
      return Number(a.segment.start) - Number(b.segment.start);
    });
  if (!visibleSegments.length) {
    subtitleOverlay.classList.remove('visible');
    subtitleOverlay.textContent = '';
    return;
  }
  const showSpeaker = document.querySelector('#showSpeaker').value === 'true';
  const useSpeakerColors = document.querySelector('#speakerColors').value === 'true';
  const fontSize = Math.max(12, Number(document.querySelector('#fontSize').value || 48));
  const marginV = Math.max(0, Number(document.querySelector('#marginV').value || 56));
  const fontName = (document.querySelector('#fontName').value || 'Noto Sans CJK SC').replace(/'/g, '');
  const outlineColor = document.querySelector('#outlineColor').value || '#000000';
  const primaryColor = document.querySelector('#primaryColor').value || '#ffffff';
  const lines = visibleSegments.map(({ segment }) => (
    showSpeaker && segment.speaker ? speakerDisplayName(segment.speaker) + ': ' + segment.text : segment.text
  ));
  const scale = assScriptScale();
  subtitleOverlay.textContent = lines.join('\\n');
  subtitleOverlay.style.fontFamily = `'${fontName}', sans-serif`;
  subtitleOverlay.style.fontSize = Math.max(10, fontSize * scale / assFontLineHeightFactor) + 'px';
  subtitleOverlay.style.lineHeight = String(assFontLineHeightFactor);
  subtitleOverlay.style.bottom = Math.max(0, marginV * scale) + 'px';
  subtitleOverlay.style.webkitTextStroke = subtitleTextStroke(scale, outlineColor);
  subtitleOverlay.style.textShadow = subtitleTextShadow(scale);
  const color = useSpeakerColors && new Set(segments.map((s) => s.speaker).filter(Boolean)).size > 1 && visibleSegments.length === 1
    ? speakerColor(visibleSegments[0].segment.speaker, segments)
    : primaryColor;
  subtitleOverlay.style.color = color;
  subtitleOverlay.style.webkitTextFillColor = color;
  subtitleOverlay.classList.add('visible');
}

function isSegmentVisibleAtTime(segment, time) {
  const start = Number(segment.start);
  const end = Number(segment.end);
  // 后端按音频能量计算句尾缓冲；旧任务/离线数据仍使用固定兜底值。
  const displayEnd = Number.isFinite(Number(segment.display_end))
    ? Number(segment.display_end)
    : end + 0.50;
  return Number.isFinite(start) && Number.isFinite(end) && start <= time && time < displayEnd;
}

function findSegmentIndexAtTime(segments, time) {
  let lo = 0;
  let hi = segments.length - 1;
  let candidate = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const start = Number(segments[mid].start);
    if (Number.isFinite(start) && start <= time) {
      candidate = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  for (let index = Math.max(0, candidate - 2); index < Math.min(segments.length, candidate + 3); index++) {
    if (isSegmentVisibleAtTime(segments[index], time)) return index;
  }
  return segments.findIndex((segment) => isSegmentVisibleAtTime(segment, time));
}

function updateSourceMaskPreview() {
  const enabled = document.querySelector('#maskEnabled').value === 'true';
  sourceMaskOverlay.classList.toggle('visible', enabled);
  maskPreviewVideo.classList.toggle('visible', false);
  if (!enabled) {
    unloadMaskPreviewSource();
    return;
  }
  const scale = assScriptScale();
  const mode = document.querySelector('#maskMode').value || 'blur';
  const height = Math.max(1, Number(document.querySelector('#maskHeight').value || 120));
  const marginV = Math.max(0, Number(document.querySelector('#maskMarginV').value || 0));
  const opacity = Math.max(0, Math.min(1, Number(document.querySelector('#maskOpacity').value || 0.82)));
  const blur = Math.max(1, Number(document.querySelector('#maskBlur').value || 24));
  const scaledHeight = Math.max(1, height * scale);
  const scaledBottom = Math.max(0, marginV * scale);
  const stageHeight = Math.max(1, videoStage.clientHeight || assPlayRes.y || 1);
  const clipTop = Math.max(0, stageHeight - scaledBottom - scaledHeight);
  const clipBottom = Math.max(0, scaledBottom);
  sourceMaskOverlay.style.height = scaledHeight + 'px';
  sourceMaskOverlay.style.bottom = scaledBottom + 'px';
  if (mode === 'bar') {
    sourceMaskOverlay.style.background = `rgba(0, 0, 0, ${opacity})`;
    unloadMaskPreviewSource();
  } else {
    if (!ensureMaskPreviewSource()) return;
    maskPreviewVideo.classList.add('visible');
    maskPreviewVideo.style.clipPath = `inset(${clipTop}px 0 ${clipBottom}px 0)`;
    maskPreviewVideo.style.filter = `blur(${Math.max(1, blur * scale)}px)`;
    sourceMaskOverlay.style.background = 'rgba(0, 0, 0, 0.18)';
    syncMaskPreviewPlayback();
  }
}

function subtitleTextStroke(scale, outlineColor = '#000') {
  return Math.max(1, 3 * scale) + 'px ' + outlineColor;
}

function subtitleTextShadow(scale) {
  const shadow = Math.max(0.5, 1 * scale);
  const blur = Math.max(1, 3 * scale);
  return `0 ${shadow}px ${blur}px rgba(0, 0, 0, 0.65)`;
}

function speakerColor(speaker, segments) {
  return speakerColorOf(speaker, (segments || []).map((segment) => segment.speaker));
}

function speakerColorOf(speaker, speakers) {
  if (speaker && speakerColorOverrides[speaker]) return speakerColorOverrides[speaker];
  const sorted = [...new Set((speakers || []).filter(Boolean))].sort();
  const index = Math.max(0, sorted.indexOf(speaker || ''));
  return speakerPalette[index % speakerPalette.length];
}

// 按说话人配色是否生效（需 >1 个说话人，与烧录/预览的回落规则一致）
function speakerColorsEnabled() {
  if (document.querySelector('#speakerColors').value !== 'true') return false;
  const rows = [...tbody.querySelectorAll('tr[data-index] .speaker')];
  const speakers = rows.map((input) => input.value.trim()).filter(Boolean);
  if (!speakers.length && cachedSegments) {
    return new Set(cachedSegments.map((s) => s.speaker).filter(Boolean)).size > 1;
  }
  return new Set(speakers).size > 1;
}

// 刷新编辑表格每行的说话人颜色点（配色模式切换/说话人改名时调用）
function refreshSpeakerDots() {
  const enabled = speakerColorsEnabled();
  tbody.querySelectorAll('tr[data-index]').forEach((tr) => {
    const dot = tr.querySelector('.speaker-dot');
    if (!dot) return;
    const speaker = (tr.querySelector('.speaker').value || '').trim();
    if (enabled && speaker) {
      const speakers = [...tbody.querySelectorAll('tr[data-index] .speaker')]
        .map((input) => input.value.trim()).filter(Boolean);
      dot.style.background = speakerColorOf(speaker, speakers);
      dot.title = '按说话人配色';
    } else {
      dot.style.background = 'transparent';
      dot.title = '';
    }
  });
}

function activeLlmProfile() {
  const profiles = (llmProfiles && llmProfiles.profiles) || [];
  return profiles.find((p) => p.id === llmProfiles.active_id) || null;
}

function updateProofreadAction() {
  const busy = currentJob && RUNNING_STATES.has(currentJob.status);
  const active = activeLlmProfile();
  proofreadRunBtn.disabled = !currentJob || busy || !active;
  proofreadRunBtn.textContent = busy && currentJob && currentJob.status === 'proofreading' ? '校对中...' : (active ? '开始校对' : '未配置 AI 服务');
  openProofreadBtn.disabled = !currentJob;
  const target = currentJob && currentJob.translation && currentJob.translation.source_available ? '英文源稿（译文不受影响，应用后需重新翻译）' : '当前字幕稿';
  proofreadModelStatusEl.textContent = active
    ? `已激活：${active.name} · ${active.model || '默认模型'}。校对目标：${target}。`
    : '未配置 AI 服务。请到 设置 → AI 服务 添加并激活一个 API 配置。';
}

function openProofreadModal() {
  updateProofreadAction();
  proofreadModal.classList.remove('is-hidden');
  if (!currentJob) return;
  const proof = (currentJob && currentJob.proofread) || {};
  if (proof.result_available && !proofreadResult) {
    loadProofreadResult();
  } else if (!proof.read_result && proofreadResult) {
    renderProofreadResult(proofreadResult);
  } else {
    updateProofreadAction();
  }
}

async function loadProofreadResult() {
  if (!currentJob) return;
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/proofread`));
    if (!res.ok) return;
    const data = await res.json();
    proofreadResult = data;
    renderProofreadResult(data);
  } catch (err) { /* ignore */ }
}

function stopProofreadPolling() {
  if (proofreadPollTimer) {
    clearInterval(proofreadPollTimer);
    proofreadPollTimer = null;
  }
}

function startProofreadPolling() {
  stopProofreadPolling();
  if (!currentJob) return;
  proofreadPollTimer = setInterval(async () => {
    if (!currentJob || !proofreadModal || proofreadModal.classList.contains('is-hidden')) {
      stopProofreadPolling();
      return;
    }
    try {
      const res = await fetch(apiUrl(`api/jobs/${currentJob.id}`));
      if (!res.ok) return;
      const job = await res.json();
      const proof = job.proofread || {};
      const percent = Number(proof.percent || 0);
      proofreadProgressMetaEl.classList.remove('is-hidden');
      proofreadProgressEl.classList.remove('is-hidden');
      proofreadProgressTextEl.textContent = `${Math.round(percent)}%`;
      proofreadProgressBarEl.style.width = `${Math.max(2, Math.min(100, percent))}%`;
      if (job.status !== 'proofreading') {
        stopProofreadPolling();
        proofreadProgressTextEl.textContent = '100%';
        proofreadProgressBarEl.style.width = '100%';
      }
    } catch (err) { /* ignore */ }
  }, 2000);
}

async function runProofread() {
  if (!currentJob) return;
  const active = activeLlmProfile();
  if (!active) {
    proofreadStatusEl.textContent = '未配置 AI 服务。请到 设置 → AI 服务 添加并激活一个 API 配置。';
    return;
  }
  const saved = await saveSegments();
  if (!saved) return;
  proofreadRunBtn.disabled = true;
  const translated = !!(currentJob.translation && currentJob.translation.source_available);
  proofreadStatusEl.textContent = translated
    ? '校对中...（源稿错字修正 + 术语分析 + 译文对照检查）'
    : '校对中...（错字修正 + 全片术语分析）';
  proofreadProgressMetaEl.classList.remove('is-hidden');
  proofreadProgressEl.classList.remove('is-hidden');
  proofreadProgressTextEl.textContent = '0%';
  proofreadProgressBarEl.style.width = '2%';
  dismissedAlignmentItems = new Set();
  currentJob = { ...currentJob, status: 'proofreading' };
  jobs = jobs.map((job) => job.id === currentJob.id ? currentJob : job);
  renderCurrentJob(currentJob, { skipSegments: true });
  ensurePolling();
  startProofreadPolling();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/proofread`), { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '校对失败');
    stopProofreadPolling();
    proofreadProgressTextEl.textContent = '100%';
    proofreadProgressBarEl.style.width = '100%';
    proofreadResult = data;
    renderProofreadResult(data);
    await refreshJobs({ keepSelection: true, skipSegments: true });
  } catch (err) {
    stopProofreadPolling();
    proofreadStatusEl.textContent = '校对失败：' + (err.message || err);
  } finally {
    updateProofreadAction();
  }
}

function renderProofreadResult(data) {
  if (!data) return;
  const suggestions = data.suggestions || [];
  const terms = data.term_corrections || [];
  const alignment = data.alignment || null;
  const applied = !!data.applied;

  const alignmentCount = alignment ? (alignment.issue_count || (alignment.issues || []).length) : 0;
  const parts = [`${suggestions.length} 处错字修正`, `${terms.length} 条术语`];
  if (alignment) parts.push(`译文对照 ${alignmentCount} 处疑似`);
  proofreadStatusEl.textContent = applied
    ? '以下为最近一次校对结果（已应用过一次，可重新勾选应用其余项）。'
    : `校对完成，耗时 ${data.elapsed_sec || 0}s：${parts.join('、')}。`;

  // ---- terms
  proofreadTermSectionEl.classList.toggle('is-hidden', !terms.length);
  proofreadTermsMetaEl.textContent = terms.length ? `${terms.length} 条，共命中 ${terms.reduce((sum, t) => sum + Number(t.hits || 0), 0)} 处` : '';
  proofreadTermsListEl.innerHTML = terms.map((t, i) => `
    <div class="proofread-item" data-term-index="${i}">
      <div class="proofread-item-head">
        <label class="proofread-check"><input type="checkbox" data-term-check="${i}" ${applied ? '' : 'checked'} /> <span class="id">${escapeHtml(t.wrong || '')}</span> → <span class="id">${escapeHtml(t.right || '')}</span></label>
        <span>命中 ${Number(t.hits || 0)} 处</span>
      </div>
      ${(t.previews || []).slice(0, 2).map((p) => `
        <div class="proofread-diff"><span class="before">${escapeHtml(p.original || '')}</span><br />→ <span class="after">${escapeHtml(p.corrected || '')}</span></div>`).join('')}
    </div>`).join('');

  // ---- typos
  proofreadTypoSectionEl.classList.toggle('is-hidden', !suggestions.length);
  proofreadTyposMetaEl.textContent = suggestions.length ? `${suggestions.length} 处` : '';
  proofreadTyposListEl.innerHTML = suggestions.map((s, i) => `
    <div class="proofread-item" data-typo-index="${i}">
      <div class="proofread-item-head">
        <label class="proofread-check"><input type="checkbox" data-typo-check="${i}" ${applied ? '' : 'checked'} /> <span class="id">${escapeHtml(s.id || '')}</span></label>
        <span>${s.type === 'typo' ? '错字/标点' : s.type}</span>
      </div>
      <div class="proofread-diff"><span class="before">${escapeHtml(s.original || '')}</span><br />→ <span class="after">${escapeHtml(s.corrected || '')}</span></div>
    </div>`).join('');

  // ---- alignment（只读标注，仅已翻译任务有）
  renderAlignmentIssues(alignment);

  if (!terms.length && !suggestions.length && !alignmentCount) {
    proofreadStatusEl.textContent = '校对完成：没有发现需要修改的地方。';
  }
  updateProofreadSelection();
}

function toggleProofreadGroup(group) {
  const master = group === 'term' ? proofreadTermsAllEl : proofreadTyposAllEl;
  const selector = group === 'term' ? '[data-term-check]' : '[data-typo-check]';
  document.querySelectorAll(selector).forEach((box) => { box.checked = master.checked; });
  document.querySelectorAll(group === 'term' ? '[data-term-index]' : '[data-typo-index]').forEach((item) => {
    item.classList.toggle('unchecked', !master.checked);
  });
  updateProofreadSelection();
}

function updateProofreadSelection() {
  const typoBoxes = Array.from(document.querySelectorAll('[data-typo-check]'));
  const termBoxes = Array.from(document.querySelectorAll('[data-term-check]'));
  const alignmentBoxes = Array.from(document.querySelectorAll('[data-alignment-check]'));
  const typoCount = typoBoxes.filter((b) => b.checked).length;
  const termCount = termBoxes.filter((b) => b.checked).length;
  const alignmentCount = alignmentBoxes.filter((b) => b.checked).length;
  const total = typoCount + termCount + alignmentCount;
  const parts = [];
  if (typoCount) parts.push(`${typoCount} 处修正`);
  if (termCount) parts.push(`${termCount} 条术语`);
  if (alignmentCount) parts.push(`${alignmentCount} 处译文`);
  proofreadSelectionMetaEl.textContent = total ? `已选 ${parts.join(' + ')}` : '未选择任何修改';
  proofreadApplyBtn.disabled = !currentJob || total === 0;
}

[proofreadTermsListEl, proofreadTyposListEl, proofreadAlignmentListEl].forEach((listEl) => {
  listEl.addEventListener('change', (event) => {
    const box = event.target;
    if (!box || box.type !== 'checkbox') return;
    const item = box.closest('.proofread-item');
    if (item) item.classList.toggle('unchecked', !box.checked);
    updateProofreadSelection();
  });
});

proofreadAlignmentAllEl.addEventListener('change', () => {
  document.querySelectorAll('[data-alignment-check]').forEach((box) => { box.checked = proofreadAlignmentAllEl.checked; });
  document.querySelectorAll('[data-alignment-id]').forEach((item) => {
    item.classList.toggle('unchecked', !proofreadAlignmentAllEl.checked);
  });
  updateProofreadSelection();
});

async function applyProofread() {
  if (!currentJob || !proofreadResult) return;
  const ids = Array.from(document.querySelectorAll('[data-typo-check]'))
    .filter((b) => b.checked)
    .map((b) => proofreadResult.suggestions[Number(b.dataset.typoCheck)].id);
  const terms = Array.from(document.querySelectorAll('[data-term-check]'))
    .filter((b) => b.checked)
    .map((b) => {
      const t = proofreadResult.term_corrections[Number(b.dataset.termCheck)];
      return { wrong: t.wrong, right: t.right };
    });
  const alignmentIds = Array.from(document.querySelectorAll('[data-alignment-check]'))
    .filter((b) => b.checked)
    .map((b) => b.dataset.alignmentCheck)
    .filter(Boolean);
  if (!ids.length && !terms.length && !alignmentIds.length) return;
  proofreadApplyBtn.disabled = true;
  proofreadStatusEl.textContent = '正在应用修改...';
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/proofread/apply`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids, terms })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '应用失败');
    let message = `已应用 ${data.applied_count || 0} 处修正`;
    if (data.term_hits) message += `、术语替换 ${data.term_hits} 处`;
    if (alignmentIds.length) {
      try {
        const alignmentRes = await fetch(apiUrl(`api/jobs/${currentJob.id}/alignment/apply`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ids: alignmentIds })
        });
        const alignmentData = await alignmentRes.json();
        if (!alignmentRes.ok) throw new Error(alignmentData.detail || '译文修正应用失败');
        if (alignmentData.applied_count) {
          message += `、译文修正 ${alignmentData.applied_count} 处`;
        }
        if (Array.isArray(alignmentData.segments) && alignmentData.segments.length) {
          renderSegments(alignmentData.segments, activeSegmentIndex >= 0 ? activeSegmentIndex : 0);
          setEditorDirty(false);
        }
        // 应用成功的条目从结果中移除,重渲染弹窗
        if (proofreadResult.alignment && Array.isArray(proofreadResult.alignment.issues)) {
          const appliedSet = new Set(alignmentData.applied_ids || []);
          proofreadResult.alignment.issues = proofreadResult.alignment.issues
            .filter((item) => !appliedSet.has(item.id));
          proofreadResult.alignment.issue_count = proofreadResult.alignment.issues.length;
        }
      } catch (alignmentErr) {
        message += `；译文修正应用失败：${alignmentErr.message || alignmentErr}`;
      }
    }
    message += '。';
    if (data.needs_retranslate) message += '本次修改写入了英文源稿，请重新翻译以同步译文。';
    proofreadStatusEl.textContent = message;
    proofreadResult = { ...proofreadResult, applied: true };
    renderProofreadResult(proofreadResult);
    await refreshJobs({ keepSelection: true, skipSegments: false });
  } catch (err) {
    proofreadStatusEl.textContent = '应用失败：' + (err.message || err);
  } finally {
    updateProofreadSelection();
    updateProofreadAction();
  }
}

// ------------------------------------------------------------ LLM profiles

async function loadLlmProfiles() {
  try {
    const res = await fetch(apiUrl('api/llm/profiles'));
    if (!res.ok) return;
    const data = await res.json();
    llmProfiles = data || { active_id: null, profiles: [] };
    renderLlmProfiles();
    updateProofreadAction();
    updateClipActions();
  } catch (err) { /* ignore */ }
}

function renderLlmProfiles() {
  const profiles = (llmProfiles && llmProfiles.profiles) || [];
  const activeId = llmProfiles && llmProfiles.active_id;
  if (!profiles.length) {
    llmProfileListEl.innerHTML = '<div class="meta">还没有 API 配置。点击"新增配置"添加一个。</div>';
    return;
  }
  llmProfileListEl.innerHTML = profiles.map((p) => `
    <div class="llm-profile-item ${p.id === activeId ? 'active' : ''}" data-profile-id="${escapeHtml(p.id)}">
      <div class="llm-profile-item-info">
        <div class="llm-profile-item-name">
          <span>${escapeHtml(p.name || '未命名')}</span>
          ${p.id === activeId ? '<span class="active-badge">使用中</span>' : ''}
        </div>
        <div class="llm-profile-item-meta">${escapeHtml(p.provider === 'ollama' ? 'Ollama' : 'OpenAI 兼容')} · ${escapeHtml(p.model || '默认模型')} · ${escapeHtml(p.base_url || '')} · ${escapeHtml(p.api_key_masked || '无 Key')}</div>
      </div>
      <div class="llm-profile-item-actions">
        ${p.id === activeId ? '' : `<button class="ghost small" type="button" data-llm-action="activate">启用</button>`}
        <button class="ghost small" type="button" data-llm-action="edit">编辑</button>
        <button class="ghost small" type="button" data-llm-action="delete">删除</button>
      </div>
    </div>`).join('');
}

function showLlmProfileEditor(profile) {
  llmEditingProfileId = profile ? profile.id : null;
  llmProfileNameInput.value = profile ? profile.name : '';
  llmProfileProviderSelect.value = profile ? (profile.provider || 'openai') : 'openai';
  llmProfileBaseUrlInput.value = profile ? profile.base_url : '';
  llmProfileModelInput.value = profile ? (profile.model || '') : '';
  llmProfileApiKeyInput.value = '';
  llmProfileApiKeyInput.placeholder = profile && profile.api_key_masked ? `当前 ${profile.api_key_masked}，留空不修改` : 'sk-...';
  llmProfileDisableThinkingSelect.value = profile && profile.disable_thinking ? 'true' : 'false';
  llmProfileTestResultEl.textContent = '';
  llmProfileEditorEl.classList.remove('is-hidden');
}

function hideLlmProfileEditor() {
  llmEditingProfileId = null;
  llmProfileEditorEl.classList.add('is-hidden');
}

async function saveLlmProfile() {
  const name = llmProfileNameInput.value.trim();
  const baseUrl = llmProfileBaseUrlInput.value.trim();
  if (!name) { llmProfileTestResultEl.textContent = '请填写名称。'; return; }
  if (!baseUrl) { llmProfileTestResultEl.textContent = '请填写 Base URL。'; return; }
  const payload = {
    name,
    base_url: baseUrl,
    model: llmProfileModelInput.value.trim(),
    provider: llmProfileProviderSelect.value,
    api_key: llmProfileApiKeyInput.value.trim(),
    disable_thinking: llmProfileDisableThinkingSelect.value === 'true',
  };
  try {
    const res = await fetch(llmEditingProfileId
      ? apiUrl(`api/llm/profiles/${llmEditingProfileId}`)
      : apiUrl('api/llm/profiles'), {
      method: llmEditingProfileId ? 'PUT' : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '保存失败');
    llmProfiles = data && data.profiles ? data : llmProfiles;
    hideLlmProfileEditor();
    await loadLlmProfiles();
    llmProfileTestResultEl.textContent = '';
  } catch (err) {
    llmProfileTestResultEl.textContent = '保存失败：' + (err.message || err);
  }
}

llmProfileListEl.addEventListener('click', async (event) => {
  const btn = event.target.closest('button[data-llm-action]');
  if (!btn) return;
  const item = btn.closest('[data-profile-id]');
  const profileId = item ? item.dataset.profileId : null;
  if (!profileId) return;
  const action = btn.dataset.llmAction;
  try {
    if (action === 'activate') {
      const res = await fetch(apiUrl(`api/llm/profiles/${profileId}/activate`), { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '切换失败');
      llmProfiles = data;
      renderLlmProfiles();
      updateProofreadAction();
      updateClipActions();
    } else if (action === 'edit') {
      const profile = (llmProfiles.profiles || []).find((p) => p.id === profileId);
      if (profile) showLlmProfileEditor(profile);
    } else if (action === 'delete') {
      if (!window.confirm('确定删除这个 API 配置？')) return;
      const res = await fetch(apiUrl(`api/llm/profiles/${profileId}`), { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '删除失败');
      llmProfiles = data;
      renderLlmProfiles();
      updateProofreadAction();
      updateClipActions();
    }
  } catch (err) {
    llmProfileTestResultEl.textContent = (err.message || err);
  }
});

async function testLlmProfile() {
  llmProfileTestResultEl.textContent = '测试中...';
  const baseUrl = llmProfileBaseUrlInput.value.trim();
  if (!baseUrl) { llmProfileTestResultEl.textContent = '请先填写 Base URL。'; return; }
  const payload = {
    name: llmProfileNameInput.value.trim() || '未命名',
    base_url: baseUrl,
    model: llmProfileModelInput.value.trim(),
    provider: llmProfileProviderSelect.value,
    api_key: llmProfileApiKeyInput.value.trim(),
    disable_thinking: llmProfileDisableThinkingSelect.value === 'true',
  };
  try {
    // 临时保存后测试（编辑态的 key 可能来自存储）
    let profileId = llmEditingProfileId;
    if (!profileId) {
      const res = await fetch(apiUrl('api/llm/profiles'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '保存失败');
      profileId = data.id;
      await loadLlmProfiles();
      hideLlmProfileEditor();
    } else {
      const res = await fetch(apiUrl(`api/llm/profiles/${profileId}`), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '保存失败');
      await loadLlmProfiles();
    }
    const res = await fetch(apiUrl('api/llm/test'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ profile_id: profileId })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '测试失败');
    llmProfileTestResultEl.textContent = data.message || (data.ok ? '连接成功' : '连接失败');
  } catch (err) {
    llmProfileTestResultEl.textContent = '测试失败：' + (err.message || err);
  }
}

async function translateCurrentSubtitles() {
  if (!currentJob || !translatorAvailable) {
    translateStatusEl.textContent = '翻译服务未启动。请用 start_ollama.bat 或 start_vllm.bat 启动。';
    return;
  }
  const saved = await saveSegments();
  if (!saved) return;
  translateZhBtn.disabled = true;
  translateStatusEl.textContent = '翻译中...';
  const total = collectSegments().length;
  currentJob = {
    ...currentJob,
    status: 'translating',
    progress: 0.95,
    error: null,
    translation: {
      ...(currentJob.translation || {}),
      in_progress: true,
      done: 0,
      total,
      percent: 0
    }
  };
  jobs = jobs.map((job) => job.id === currentJob.id ? currentJob : job);
  renderCurrentJob(currentJob, { skipSegments: true });
  ensurePolling();
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/translate`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_language: targetLanguageInput.value || '简体中文',
        mode: translateModeSelect.value || 'bilingual',
        protected_terms: translateProtectedTermsInput.value || ''
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '翻译失败');
    currentJob = {
      ...currentJob,
      translation: {
        ...(currentJob.translation || {}),
        validation_issue_count: data.validation_issue_count || 0,
        validation_issues: data.validation_issues || [],
        pretranslation_skip_count: data.pretranslation_skip_count || 0,
        pretranslation_skips: data.pretranslation_skips || [],
      }
    };
    renderSegments(data.segments || [], activeSegmentIndex >= 0 ? activeSegmentIndex : 0);
    setEditorDirty(false);
    translateStatusEl.textContent = translationDoneStatus(data);
    renderTranslationReview(data);
    await refreshJobs({ keepSelection: true, skipSegments: true });
  } catch (err) {
    translateStatusEl.textContent = '翻译失败：' + (err.message || err);
  } finally {
    updateTranslateAction();
  }
}

function updateTranslateAction() {
  const busy = currentJob && RUNNING_STATES.has(currentJob.status);
  translateZhBtn.disabled = !translatorAvailable || !currentJob || busy;
  translateZhBtn.textContent = translatorAvailable ? '开始翻译' : '翻译模型未启动';
  openTranslateBtn.disabled = !currentJob;
  const model = translatorInfo.model || 'Qwen2.5-3B-Instruct-AWQ';
  translateModelStatusEl.textContent = translatorAvailable
    ? `已配置本地模型：${model}。翻译前会自动保留英文底稿。`
    : '未配置本地翻译模型。请在前台运行 start_ollama.bat 或 start_vllm.bat 后重新打开工作台。';
  const translation = (currentJob && currentJob.translation) || {};
  restoreTranslationBtn.disabled = !currentJob || !translation.source_available || busy;
  updateTranslateProgress(currentJob);
  renderTranslationReview(currentJob);
}

function translationDoneStatus(data) {
  const count = Number(data.count || 0);
  const issueCount = Number(data.validation_issue_count || 0);
  const skipCount = Number(data.pretranslation_skip_count || 0);
  const extras = [];
  if (skipCount) extras.push(`自动跳过 ${skipCount} 条`);
  if (issueCount) extras.push(`可疑 ${issueCount} 条`);
  return '已翻译 ' + count + ' 条字幕' + (extras.length ? '，' + extras.join('，') : '') + '。';
}

async function restoreSourceSubtitles() {
  if (!currentJob) return;
  restoreTranslationBtn.disabled = true;
  translateStatusEl.textContent = '正在恢复翻译前字幕...';
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/translate/restore`), { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '恢复失败');
    renderSegments(data.segments || [], 0);
    setEditorDirty(false);
    currentJob = {
      ...currentJob,
      translation: {
        ...(currentJob.translation || {}),
        validation_issue_count: 0,
        validation_issues: [],
        pretranslation_skip_count: 0,
        pretranslation_skips: [],
      }
    };
    dismissedTranslationReviewItems = new Set();
    renderTranslationReview(currentJob);
    translateStatusEl.textContent = `已恢复 ${data.count || 0} 条翻译前字幕。`;
    await refreshJobs({ keepSelection: true, skipSegments: true });
  } catch (err) {
    translateStatusEl.textContent = '恢复失败：' + (err.message || err);
  } finally {
    updateTranslateAction();
  }
}

// ------------------------------------------------------------ 对照检查

function alignmentTypeLabel(type) {
  const labels = {
    omission: '疑似漏译',
    addition: '疑似加译',
    mistranslation: '疑似误译',
    terminology: '术语不一致',
  };
  return labels[type] || '待检查';
}

function alignmentItemKey(item) {
  return [item.type || 'alignment', item.id || '', item.index == null ? '' : item.index].join(':');
}

function renderAlignmentIssues(alignment) {
  const items = (alignment && Array.isArray(alignment.issues) ? alignment.issues : [])
    .filter((item) => !dismissedAlignmentItems.has(alignmentItemKey(item)));
  proofreadAlignmentSectionEl.classList.toggle('is-hidden', !items.length);
  if (!items.length) {
    proofreadAlignmentMetaEl.textContent = '';
    proofreadAlignmentListEl.innerHTML = '';
    return;
  }
  const selectable = items.filter((item) => item.suggested && item.suggested !== item.translated_text);
  proofreadAlignmentMetaEl.textContent = `已检查 ${alignment.pair_count || 0} 对 · ${items.length} 处疑似`;
  proofreadAlignmentListEl.innerHTML = items.map((item) => {
    const index = Number(item.index);
    const start = Number(item.start || 0);
    const key = alignmentItemKey(item);
    const severity = item.type === 'addition' || item.type === 'terminology' ? 'info' : 'warning';
    const canApply = item.suggested && item.suggested !== item.translated_text;
    const checkRow = canApply
      ? `<label class="proofread-check"><input type="checkbox" data-alignment-check="${escapeHtml(item.id || '')}" checked /></label>`
      : '';
    const suggestionRow = canApply
      ? `<div class="proofread-diff"><span class="before">${escapeHtml(item.translated_text || '')}</span><br />→ <span class="after">${escapeHtml(item.suggested || '')}</span></div>`
      : `<div class="proofread-diff"><span class="before">${escapeHtml(item.translated_text || '')}</span></div>`;
    return `
      <div class="proofread-item ${severity}" data-index="${Number.isFinite(index) ? index : -1}" data-start="${Number.isFinite(start) ? start : 0}" data-key="${escapeHtml(key)}" data-alignment-id="${escapeHtml(item.id || '')}">
        <div class="proofread-item-head">
          ${checkRow}
          <span class="id">${escapeHtml(alignmentTypeLabel(item.type))} · ${Number.isFinite(index) ? '#' + (index + 1) : '?'} · ${formatTimelineTime(start)}</span>
          <strong>${escapeHtml(item.note || '')}</strong>
        </div>
        <div class="translation-review-text">SRC ${escapeHtml(item.source_text || '')}</div>
        ${suggestionRow}
        <div class="translation-review-actions">
          <button class="ghost small" type="button" data-alignment-action="jump">跳到字幕</button>
          <button class="ghost small" type="button" data-alignment-action="play">回看原片</button>
          <button class="primary small" type="button" data-alignment-action="dismiss">标为已处理</button>
        </div>
      </div>`;
  }).join('');
  updateProofreadSelection();
}

function useActiveSegmentAsClipRange() {
  const segments = collectSegments();
  const segment = segments[activeSegmentIndex];
  if (!segment) {
    clipStatusEl.textContent = '先选中一行字幕。';
    return;
  }
  const clip = activeClip() || addClipToQueue({
    id: 'manual',
    start: segment.start,
    end: segment.end,
    title: segment.text || '当前字幕切片',
    reason: 'manual range from selected subtitle',
    score: 0,
    selection_method: 'manual'
  }, { silent: true });
  updateClipFromValues(clip.id, { start: segment.start, end: segment.end, seek: true });
  clipStatusEl.textContent = '已把当前字幕时间写入当前切片。';
}

async function findClipCandidates(strategy = 'model') {
  if (!currentJob) return;
  if (strategy === 'model' && !activeLlmProfile()) {
    clipStatusEl.textContent = 'AI 精选需要先在首页配置并启用一个 AI 服务；也可以先用规则粗筛。';
    return;
  }
  const saved = await saveSegments();
  if (!saved) return;
  findClipsBtn.disabled = true;
  findClipsRulesBtn.disabled = true;
  clipStatusEl.textContent = strategy === 'model' ? '正在生成候选并让模型评选...' : '正在按结构和时长粗筛...';
  try {
    const minDuration = Math.max(10, Number(clipMinDurationInput.value || 60));
    const targetDuration = Math.max(minDuration, Number(clipTargetDurationInput.value || 120));
    // 最长秒数留空 = 不设上限，长度由目标秒数评分和 AI 判断
    const rawMax = Number(clipMaxDurationInput.value);
    const maxDuration = clipMaxDurationInput.value.trim() === '' || !Number.isFinite(rawMax) || rawMax <= 0
      ? 0
      : Math.max(targetDuration, rawMax);
    const limit = strategy === 'model' ? 8 : 24;
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/clips?min_duration=${minDuration}&target_duration=${targetDuration}&max_duration=${maxDuration}&limit=${limit}&strategy=${strategy}`), { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '查找候选片段失败');
    renderClipCandidates(data.clips || []);
    clipStatusEl.textContent = (data.clips || []).length
      ? (strategy === 'model' ? 'AI 精选完成。仍建议回看原片并微调边界。' : '规则粗筛完成，这些不是模型判断结果。')
      : '没有找到合适候选片段。';
  } catch (err) {
    clipStatusEl.textContent = '查找候选片段失败：' + (err.message || err);
  } finally {
    findClipsBtn.disabled = false;
    findClipsRulesBtn.disabled = false;
    updateClipActions();
  }
}

function updateClipActions() {
  openClipsBtn.disabled = !currentJob;
  findClipsBtn.disabled = !activeLlmProfile() || !currentJob;
  findClipsRulesBtn.disabled = !currentJob;
  renderClipBtn.disabled = !currentJob || !ffmpegAvailable || !activeClip();
  renderClipQueueBtn.disabled = !currentJob || !ffmpegAvailable || !selectedClips.length;
  const active = activeLlmProfile();
  clipModelStatusEl.textContent = active
    ? `AI 精选使用 ${active.model || '默认模型'}（${active.provider === 'ollama' ? 'Ollama' : 'OpenAI 兼容'}）；规则粗筛只生成候选，不代表内容质量。`
    : 'AI 精选未启用（首页未配置 AI 服务）；当前只能规则粗筛。';
  if (clipQueueCountEl) clipQueueCountEl.textContent = `${selectedClips.length} 个`;
}

function renderClipCandidates(clips) {
  if (clipCandidateCountEl) clipCandidateCountEl.textContent = `${clips.length} 个`;
  if (!clips.length) {
    clipListEl.innerHTML = '<div class="clip-empty">没有候选片段</div>';
    return;
  }
  clipListEl.innerHTML = clips.map((clip) => {
    const encoded = encodeClipPayload(clip);
    return `
    <div class="clip-card" data-clip-start="${Number(clip.start) || 0}" data-clip-end="${Number(clip.end) || 0}" data-clip-payload="${encoded}">
      <div class="clip-card-head">
        <span>原片 ${formatTimelineTime(clip.start)} - ${formatTimelineTime(clip.end)} · ${Math.round(Number(clip.duration) || 0)}s</span>
        <strong>${clip.selection_method === 'model' ? 'AI ' : '规则 '}${Math.round(Number(clip.score) || 0)}</strong>
      </div>
      <div class="clip-title">${escapeHtml(clip.title || '未命名片段')}</div>
      <div class="clip-reason">${escapeHtml(clip.reason || '')}</div>
      <div class="clip-actions" style="margin-top:8px">
        <button class="ghost small preview-clip" type="button">回看原片</button>
        <button class="primary small pick-clip" type="button">选用并微调</button>
      </div>
    </div>
  `;
  }).join('');
}

clipListEl.addEventListener('click', (event) => {
  const button = event.target.closest('.pick-clip, .preview-clip');
  if (!button) return;
  const card = button.closest('.clip-card');
  if (!card) return;
  const start = Number(card.dataset.clipStart || 0);
  if (button.classList.contains('preview-clip')) {
    preview.currentTime = start;
    preview.play().catch(() => {});
    return;
  }
  clipListEl.querySelectorAll('.clip-card').forEach((item) => item.classList.toggle('selected', item === card));
  const clip = decodeClipPayload(card.dataset.clipPayload) || {
    start,
    end: Number(card.dataset.clipEnd || 0),
    title: '未命名片段',
    reason: '',
    score: 0,
    selection_method: 'rules'
  };
  addClipToQueue(clip);
});

clipQueueListEl.addEventListener('click', (event) => {
  const button = event.target.closest('[data-clip-action]');
  const card = event.target.closest('.clip-card.queue');
  if (!card) return;
  const id = card.dataset.clipId;
  if (!id) return;
  if (!button) {
    selectClip(id, { seek: true });
    return;
  }
  event.stopPropagation();
  const action = button.dataset.clipAction;
  if (action === 'select') selectClip(id, { seek: true });
  if (action === 'up') moveQueuedClip(id, -1);
  if (action === 'down') moveQueuedClip(id, 1);
  if (action === 'remove') removeQueuedClip(id);
});

function ensureClipQueueForJob() {
  const jobId = currentJob ? currentJob.id : '';
  if (clipQueueJobId === jobId) return;
  clipQueueJobId = jobId;
  selectedClips = [];
  activeClipId = '';
  renderClipQueue();
  syncClipEditor();
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
    selectionMethod: source.selection_method || source.selectionMethod || 'rules'
  };
}

function activeClip() {
  return selectedClips.find((clip) => clip.id === activeClipId) || null;
}

function addClipToQueue(source, options = {}) {
  const clip = normalizeQueuedClip(source);
  selectedClips.push(clip);
  selectClip(clip.id, { seek: true });
  if (!options.silent) {
    clipStatusEl.textContent = `已加入已选切片：${formatTimelineTime(clip.start)} - ${formatTimelineTime(clip.end)}。可以改时间、改顺序再导出。`;
  }
  return clip;
}

function selectClip(id, options = {}) {
  activeClipId = id;
  const clip = activeClip();
  renderClipQueue();
  syncClipEditor();
  if (clip) setClipRange(clip.start, clip.end, options.seek);
  updateClipActions();
}

function moveQueuedClip(id, delta) {
  const index = selectedClips.findIndex((clip) => clip.id === id);
  const next = index + delta;
  if (index < 0 || next < 0 || next >= selectedClips.length) return;
  const [clip] = selectedClips.splice(index, 1);
  selectedClips.splice(next, 0, clip);
  activeClipId = id;
  renderClipQueue();
}

function removeQueuedClip(id) {
  const index = selectedClips.findIndex((clip) => clip.id === id);
  if (index < 0) return;
  selectedClips.splice(index, 1);
  if (activeClipId === id) {
    activeClipId = selectedClips[Math.min(index, selectedClips.length - 1)]?.id || '';
  }
  renderClipQueue();
  syncClipEditor();
  const clip = activeClip();
  if (clip) setClipRange(clip.start, clip.end, false);
  else updateTimelineClipRange();
  updateClipActions();
}

function renderClipQueue() {
  if (clipQueueCountEl) clipQueueCountEl.textContent = `${selectedClips.length} 个`;
  if (!selectedClips.length) {
    clipQueueListEl.innerHTML = '<div class="clip-empty">从左侧候选点“选用并微调”，或用当前字幕创建一个切片。</div>';
    return;
  }
  clipQueueListEl.innerHTML = selectedClips.map((clip, index) => `
    <div class="clip-card queue${clip.id === activeClipId ? ' selected' : ''}" data-clip-id="${escapeHtml(clip.id)}">
      <div class="clip-order">${index + 1}</div>
      <div>
        <div class="clip-card-head">
          <span>${formatTimelineTime(clip.start)} - ${formatTimelineTime(clip.end)} · ${Math.round((clip.end - clip.start) * 10) / 10}s</span>
          <strong>${clip.selectionMethod === 'model' ? 'AI ' : clip.selectionMethod === 'manual' ? '手动' : '规则 '}${clip.score ? Math.round(clip.score) : ''}</strong>
        </div>
        <div class="clip-title">${escapeHtml(clip.title)}</div>
        <div class="clip-reason">${escapeHtml(clip.reason || '手动微调')}</div>
      </div>
      <div class="clip-card-actions">
        <button class="ghost small" type="button" data-clip-action="up" title="上移">↑</button>
        <button class="ghost small" type="button" data-clip-action="down" title="下移">↓</button>
        <button class="ghost small" type="button" data-clip-action="select" title="选中">◎</button>
        <button class="ghost small" type="button" data-clip-action="remove" title="移除">×</button>
      </div>
    </div>
  `).join('');
}

function syncClipEditor() {
  const clip = activeClip();
  const disabled = !clip;
  [clipTitleInput, clipStartInput, clipEndInput, clipDurationInput, clipMoveBackBtn, clipMoveForwardBtn, clipStartEarlierBtn, clipEndLaterBtn].forEach((el) => {
    if (el) el.disabled = disabled;
  });
  if (!clip) {
    clipTitleInput.value = '';
    clipStartInput.value = '0';
    clipEndInput.value = '0';
    clipDurationInput.value = '';
    return;
  }
  clipTitleInput.value = clip.title;
  clipStartInput.value = clip.start.toFixed(1);
  clipEndInput.value = clip.end.toFixed(1);
  clipDurationInput.value = Math.max(0.25, clip.end - clip.start).toFixed(1);
}

function updateClipFromValues(id, values = {}) {
  const clip = selectedClips.find((item) => item.id === id);
  if (!clip) return;
  let start = values.start == null ? Number(clipStartInput.value || clip.start) : Number(values.start);
  let end = values.end == null ? Number(clipEndInput.value || clip.end) : Number(values.end);
  start = Math.max(0, Number.isFinite(start) ? start : clip.start);
  end = Math.max(start + 0.25, Number.isFinite(end) ? end : clip.end);
  clip.start = Math.round(start * 10) / 10;
  clip.end = Math.round(end * 10) / 10;
  if (values.title != null) clip.title = String(values.title).trim() || '未命名片段';
  activeClipId = id;
  syncClipEditor();
  renderClipQueue();
  setClipRange(clip.start, clip.end, values.seek);
  updateClipActions();
}

function nudgeActiveClip({ shift = 0, startDelta = 0, endDelta = 0 }) {
  const clip = activeClip();
  if (!clip) return;
  updateClipFromValues(clip.id, {
    start: Math.max(0, clip.start + shift + startDelta),
    end: Math.max(clip.start + shift + startDelta + 0.25, clip.end + shift + endDelta),
    seek: true
  });
}

function setClipRange(start, end, seek) {
  start = Math.max(0, Number(start) || 0);
  end = Math.max(start + 0.25, Number(end) || start + 120);
  clipStartInput.value = start.toFixed(1);
  clipEndInput.value = end.toFixed(1);
  clipDurationInput.value = (end - start).toFixed(1);
  updateTimelineClipRange();
  if (seek) preview.currentTime = start;
}

async function renderSelectedClip() {
  if (!currentJob || !ffmpegAvailable) return;
  const clip = activeClip();
  if (!clip) {
    clipStatusEl.textContent = '先从已选切片里选中一个片段。';
    return;
  }
  const saved = await saveSegments();
  if (!saved) return;
  const start = Number(clip.start || 0);
  const end = Number(clip.end || 0);
  if (!(end > start)) {
    clipStatusEl.textContent = '结束时间必须大于开始时间。';
    return;
  }
  renderClipBtn.disabled = true;
  clipStatusEl.textContent = '正在导出切片...';
  try {
    const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/clips/render`), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start,
        end,
        style: collectSubtitleStyle(),
        name: clip.title || `clip_${start.toFixed(1)}_${end.toFixed(1)}`
      })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '导出切片失败');
    const files = data.files || { mp4: data.filename };
    const links = Object.entries(files).map(([kind, filename]) => {
      const label = kind === 'metadata' ? '原片时间映射 JSON' : kind.toUpperCase();
      const href = apiUrl(`api/jobs/${currentJob.id}/clips/${encodeURIComponent(filename)}`);
      return `<a href="${href}" target="_blank">${escapeHtml(label)}</a>`;
    }).join(' · ');
    clipStatusEl.innerHTML = `已导出原片 ${formatTimelineTime(data.start)} - ${formatTimelineTime(data.end)}；切片内时间从 00:00 开始：${links}`;
  } catch (err) {
    clipStatusEl.textContent = '导出切片失败：' + (err.message || err);
  } finally {
    updateClipActions();
  }
}

async function renderQueuedClips() {
  if (!currentJob || !ffmpegAvailable || !selectedClips.length) return;
  const saved = await saveSegments();
  if (!saved) return;
  renderClipBtn.disabled = true;
  renderClipQueueBtn.disabled = true;
  const outputs = [];
  try {
    for (let index = 0; index < selectedClips.length; index += 1) {
      const clip = selectedClips[index];
      activeClipId = clip.id;
      renderClipQueue();
      setClipRange(clip.start, clip.end, true);
      clipStatusEl.textContent = `正在导出 ${index + 1}/${selectedClips.length}：${formatTimelineTime(clip.start)} - ${formatTimelineTime(clip.end)}...`;
      const res = await fetch(apiUrl(`api/jobs/${currentJob.id}/clips/render`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: clip.start,
          end: clip.end,
          style: collectSubtitleStyle(),
          name: `${String(index + 1).padStart(2, '0')}_${clip.title || `clip_${clip.start.toFixed(1)}_${clip.end.toFixed(1)}`}`
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '导出切片失败');
      outputs.push(data);
    }
    const links = outputs.map((data, index) => {
      const filename = data.filename || (data.files || {}).mp4;
      const href = apiUrl(`api/jobs/${currentJob.id}/clips/${encodeURIComponent(filename)}`);
      return `<a href="${href}" target="_blank">#${index + 1} MP4</a>`;
    }).join(' · ');
    clipStatusEl.innerHTML = `已按当前顺序导出 ${outputs.length} 个切片：${links}`;
  } catch (err) {
    clipStatusEl.textContent = '批量导出失败：' + (err.message || err);
  } finally {
    updateClipActions();
  }
}

function jobSummary(job) {
  const inference = job.inference || {};
  if (job.transcript_source && String(job.transcript_source).indexOf('captions:') === 0) {
    const parts = String(job.transcript_source).split(':');
    const kind = parts[1] === 'auto' ? '自动' : '人工';
    return '平台字幕（' + kind + (parts[2] ? ' ' + parts[2] : '') + '）· 已跳过转录';
  }
  if (job.status === 'translating') return translationProgressSummary(job);
  if (isWhisperJob(job)) return tokenUsageSummary(job);
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return tokenUsageSummary(job) + ' · max_len ' + inference.max_length + ' · ' + inference.decoding + temp;
}

function parameterSummary(job) {
  const inference = job.inference || {};
  const hotwords = job.hotwords ? (' · 热词 ' + job.hotwords.split(/[\\s,，;；]+/).filter(Boolean).length) : '';
  if (isWhisperJob(job)) return 'Whisper backend' + speakerLabelSummary(job) + hotwords;
  const temp = inference.temperature ? (' · temp ' + inference.temperature) : '';
  return 'max_len ' + inference.max_length + ' · ' + inference.decoding + temp + hotwords;
}

function tokenUsageSummary(job) {
  const usage = job.usage || {};
  const inference = job.inference || {};
  if (isWhisperJob(job)) {
    const elapsed = job.elapsed_sec == null ? elapsedJobSeconds(job) : Number(job.elapsed_sec || 0);
    const elapsedText = elapsed > 0 ? ' · ' + formatDuration(elapsed) : '';
    if (usage.generated_tokens == null) return 'Whisper 转写' + elapsedText;
    return 'Whisper 已返回 ' + usage.generated_tokens + ' 段' + elapsedText + speakerLabelSummary(job);
  }
  const maxNewTokens = usage.max_new_tokens || inference.max_new_tokens || 0;
  if (usage.generated_tokens == null) return '生成 tokens ' + maxNewTokens;
  const prompt = usage.prompt_tokens == null ? '' : (' · prompt ' + usage.prompt_tokens);
  return '生成 ' + usage.generated_tokens + '/' + maxNewTokens + ' tokens' + prompt;
}

function speakerLabelSummary(job) {
  const info = job.speaker_labeling || {};
  if (!info.enabled) return '';
  const backend = info.method ? ' · ' + info.method : (job.diarization_backend ? ' · ' + job.diarization_backend : '');
  const fallback = info.fallback ? ' · fallback' : '';
  if (info.applied && info.speakers) return ' · speakers ' + info.speakers + backend + fallback;
  if (info.reason) return ' · speakers pending' + backend;
  return '';
}

function truncationWarning(job) {
  const usage = job.usage || {};
  if (isWhisperJob(job)) return '';
  if (!usage.possibly_truncated) return '';
  return '可能截断：生成 token 已达到上限，请检查字幕末尾或提高输出 tokens 后重跑。';
}

function isWhisperJob(job) {
  return job && job.backend === 'whisper';
}

function elapsedJobSeconds(job) {
  if (!job || !job.created_at) return 0;
  return Math.max(0, (Date.now() / 1000) - Number(job.created_at || 0));
}

function formatDuration(seconds) {
  seconds = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (minutes <= 0) return rest + 's';
  return minutes + 'm ' + String(rest).padStart(2, '0') + 's';
}

function statusClass(status) {
  return 'pill ' + (status === 'failed' ? 'bad' : status === 'done' ? 'ok' : status === 'cancelled' ? 'muted' : '');
}

function statusLabel(status) {
  const labels = {
    queued: '排队中',
    downloading: '下载中',
    loading_model: '加载模型',
    transcribing: '转写中',
    postprocessing: '处理中',
    labeling_speakers: '标记说话人',
    translating: '翻译中',
    proofreading: 'AI 校对中',
    waiting_review: '待校对',
    rendering: '烧录中',
    done: '已完成',
    failed: '失败',
    cancelled: '已取消',
    idle: '空闲'
  };
  return labels[status] || status;
}

function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

refreshRuntime();
refreshJobs();
loadLlmProfiles();
loadHotwordsGlossary();

// 撤销/重做快捷键：Ctrl+Z 撤销，Ctrl+Y 或 Ctrl+Shift+Z 重做
// 在 textarea/input 内不拦截，让浏览器原生文本撤销工作
document.addEventListener('keydown', (e) => {
  const isMod = e.ctrlKey || e.metaKey;
  if (!isMod) return;
  const tag = e.target.tagName;
  if (tag === 'TEXTAREA' || tag === 'INPUT') return;
  if (e.key === 'z' && !e.shiftKey) {
    e.preventDefault();
    undoEdit();
  } else if (e.key === 'y' || (e.key === 'z' && e.shiftKey)) {
    e.preventDefault();
    redoEdit();
  }
});
