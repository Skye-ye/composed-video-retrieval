const API_BASE = window.APP_CONFIG?.API_BASE ?? "";
const USE_MOCK = new URLSearchParams(window.location.search).get("mock") === "1";
const PAGE_LIMIT = 24;
const MAX_TOP_K = 50;

const state = {
  videos: [],
  nextCursor: null,
  loadingVideos: false,
  videoError: "",
  selectedVideoId: null,
  modificationText: "",
  retainText: "",
  excludeText: "",
  advancedOpen: false,
  topK: 20,
  debug: false,
  results: [],
  queryId: "",
  searching: false,
  searchError: "",
  hasSearched: false,
  formWarning: "",
  searchRequestId: 0,
};

const els = {
  debugToggle: document.querySelector("#debugToggle"),
  videoCount: document.querySelector("#videoCount"),
  selectedPreview: document.querySelector("#selectedPreview"),
  videoGrid: document.querySelector("#videoGrid"),
  videoStatus: document.querySelector("#videoStatus"),
  loadMoreBtn: document.querySelector("#loadMoreBtn"),
  queryIdBadge: document.querySelector("#queryIdBadge"),
  resultNotice: document.querySelector("#resultNotice"),
  resultsArea: document.querySelector("#resultsArea"),
  searchForm: document.querySelector("#searchForm"),
  modificationText: document.querySelector("#modificationText"),
  retainText: document.querySelector("#retainText"),
  excludeText: document.querySelector("#excludeText"),
  topKInput: document.querySelector("#topKInput"),
  advancedToggle: document.querySelector("#advancedToggle"),
  advancedFields: document.querySelector("#advancedFields"),
  searchBtn: document.querySelector("#searchBtn"),
  formHint: document.querySelector("#formHint"),
  previewDialog: document.querySelector("#previewDialog"),
  closePreviewBtn: document.querySelector("#closePreviewBtn"),
  dialogMedia: document.querySelector("#dialogMedia"),
  dialogId: document.querySelector("#dialogId"),
  dialogTitle: document.querySelector("#dialogTitle"),
  skeletonTemplate: document.querySelector("#skeletonTemplate"),
};

const mockVideos = createMockVideos(54);

class ApiError extends Error {
  constructor(message, code, status) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

function init() {
  bindEvents();
  renderAll();
  loadVideos();
}

function bindEvents() {
  els.loadMoreBtn.addEventListener("click", () => loadVideos());

  els.debugToggle.addEventListener("change", (event) => {
    state.debug = event.target.checked;
    renderResults();
  });

  els.modificationText.addEventListener("input", (event) => {
    state.modificationText = event.target.value;
    state.formWarning = "";
    renderFormState();
  });

  els.retainText.addEventListener("input", (event) => {
    state.retainText = event.target.value;
  });

  els.excludeText.addEventListener("input", (event) => {
    state.excludeText = event.target.value;
  });

  els.topKInput.addEventListener("input", (event) => {
    state.topK = clampNumber(Number(event.target.value), 1, MAX_TOP_K, 20);
    renderFormState();
  });

  els.topKInput.addEventListener("blur", () => {
    els.topKInput.value = String(state.topK);
  });

  els.advancedToggle.addEventListener("click", () => {
    state.advancedOpen = !state.advancedOpen;
    renderAdvancedState();
  });

  els.searchForm.addEventListener("submit", handleSearchSubmit);
  els.closePreviewBtn.addEventListener("click", closePreview);

  els.previewDialog.addEventListener("click", (event) => {
    if (event.target === els.previewDialog) {
      closePreview();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && els.previewDialog.open) {
      closePreview();
    }
  });
}

function renderAll() {
  renderVideoLibrary();
  renderSelectedPreview();
  renderAdvancedState();
  renderFormState();
  renderResults();
}

async function loadVideos() {
  if (state.loadingVideos) return;

  state.loadingVideos = true;
  state.videoError = "";
  renderVideoLibrary();

  try {
    const data = await api.listVideos({
      cursor: state.nextCursor,
      limit: PAGE_LIMIT,
    });

    const videos = Array.isArray(data.videos) ? data.videos : [];
    state.videos = [...state.videos, ...videos];
    state.nextCursor = data.next_cursor || null;

  } catch (error) {
    state.videoError = getErrorMessage(error, "视频库加载失败");
  } finally {
    state.loadingVideos = false;
    renderVideoLibrary();
    renderSelectedPreview();
    renderFormState();
  }
}

async function handleSearchSubmit(event) {
  event.preventDefault();

  const invalidReason = getSearchInvalidReason();
  if (invalidReason) {
    state.formWarning = invalidReason;
    renderFormState();
    return;
  }

  const currentRequestId = ++state.searchRequestId;
  state.searching = true;
  state.searchError = "";
  state.formWarning = "";
  state.hasSearched = true;
  renderResults();
  renderFormState();

  const payload = {
    reference_video_id: state.selectedVideoId,
    modification_text: state.modificationText.trim(),
    retain_text: state.advancedOpen ? normalizeOptionalText(state.retainText) : null,
    exclude_text: state.advancedOpen ? normalizeOptionalText(state.excludeText) : null,
    top_k: state.topK,
    debug: state.debug,
  };

  try {
    const data = await api.search(payload);
    if (currentRequestId !== state.searchRequestId) return;

    state.queryId = data.query_id || "";
    state.results = Array.isArray(data.results) ? data.results : [];
  } catch (error) {
    if (currentRequestId !== state.searchRequestId) return;
    state.searchError = getErrorMessage(error, "检索请求失败");
  } finally {
    if (currentRequestId !== state.searchRequestId) return;
    state.searching = false;
    renderResults();
    renderFormState();
  }
}

function renderVideoLibrary() {
  els.videoCount.textContent = `${state.videos.length} 条`;
  els.videoGrid.replaceChildren();

  for (const video of state.videos) {
    els.videoGrid.append(createVideoCard(video));
  }

  if (state.loadingVideos && state.videos.length === 0) {
    for (let index = 0; index < 12; index += 1) {
      els.videoGrid.append(createVideoSkeleton());
    }
  }

  if (state.videoError) {
    els.videoStatus.textContent = state.videoError;
  } else if (state.loadingVideos) {
    els.videoStatus.textContent = "正在加载视频库...";
  } else if (state.nextCursor) {
    els.videoStatus.textContent = "还有更多视频可浏览";
  } else if (state.videos.length > 0) {
    els.videoStatus.textContent = "已加载全部可用视频";
  } else {
    els.videoStatus.textContent = "暂无视频数据";
  }

  els.loadMoreBtn.disabled = state.loadingVideos || !state.nextCursor;
  els.loadMoreBtn.textContent = state.loadingVideos ? "加载中" : "加载更多";
}

function renderSelectedPreview() {
  const selected = getSelectedVideo();
  els.selectedPreview.replaceChildren();
  els.selectedPreview.classList.toggle("is-empty", !selected);

  const previewFrame = document.createElement("div");
  previewFrame.className = "preview-frame";

  if (selected?.thumbnail_url) {
    const img = createImage(selected.thumbnail_url, selected.title || selected.video_id);
    previewFrame.append(img);
  }

  const copy = document.createElement("div");
  const label = document.createElement("p");
  label.className = "muted";
  label.textContent = "当前参考视频";

  const title = document.createElement("strong");
  title.textContent = selected ? getVideoTitle(selected) : "尚未选择";

  copy.append(label, title);
  els.selectedPreview.append(previewFrame, copy);
}

function renderAdvancedState() {
  els.advancedToggle.setAttribute("aria-expanded", String(state.advancedOpen));
  els.advancedFields.hidden = !state.advancedOpen;
}

function renderFormState() {
  const invalidReason = getSearchInvalidReason();
  const hint = state.formWarning || invalidReason || "准备就绪，可以发起组合检索。";

  els.searchBtn.disabled = Boolean(invalidReason) || state.searching;
  els.searchBtn.textContent = state.searching ? "检索中" : "开始检索";
  els.formHint.textContent = hint;
  els.formHint.classList.toggle("is-warning", Boolean(state.formWarning || invalidReason));
}

function renderResults() {
  const hasResults = state.results.length > 0;
  const allLowConfidence =
    hasResults && state.results.every((result) => result.low_confidence === true);

  els.queryIdBadge.textContent = state.queryId ? `Query ${state.queryId}` : "等待检索";
  els.queryIdBadge.classList.toggle("is-muted", !state.queryId);

  if (allLowConfidence) {
    els.resultNotice.hidden = false;
    els.resultNotice.textContent = "视频库中可能没有完全匹配的结果，以下是最接近的几条。";
  } else if (state.searchError) {
    els.resultNotice.hidden = false;
    els.resultNotice.textContent = state.searchError;
  } else {
    els.resultNotice.hidden = true;
    els.resultNotice.textContent = "";
  }

  els.resultsArea.classList.toggle("has-overlay", state.searching && hasResults);

  if (state.searching && !hasResults) {
    renderResultSkeletons();
    return;
  }

  if (state.searchError && !hasResults) {
    renderErrorState(state.searchError);
    return;
  }

  if (!hasResults) {
    renderEmptyResults();
    return;
  }

  els.resultsArea.replaceChildren();
  state.results.forEach((result, index) => {
    els.resultsArea.append(createResultCard(result, index + 1));
  });
}

function renderResultSkeletons() {
  els.resultsArea.replaceChildren();
  for (let index = 0; index < 8; index += 1) {
    els.resultsArea.append(els.skeletonTemplate.content.firstElementChild.cloneNode(true));
  }
}

function renderEmptyResults() {
  const empty = document.createElement("div");
  empty.className = "empty-state";

  const icon = document.createElement("div");
  icon.className = "empty-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = state.hasSearched ? "0" : "⌁";

  const title = document.createElement("h3");
  title.textContent = state.hasSearched ? "后端未返回结果" : "选择参考视频后开始检索";

  const text = document.createElement("p");
  text.textContent = state.hasSearched
    ? "可以调整修改文本或增加返回数量后再次检索。"
    : "输入自然语言修改指令，系统会返回与组合语义最接近的视频。";

  empty.append(icon, title, text);
  els.resultsArea.replaceChildren(empty);
}

function renderErrorState(message) {
  const box = document.createElement("div");
  box.className = "error-state";

  const title = document.createElement("h3");
  title.textContent = "请求失败";

  const text = document.createElement("p");
  text.textContent = message;

  box.append(title, text);
  els.resultsArea.replaceChildren(box);
}

function createVideoCard(video) {
  const button = document.createElement("button");
  button.className = "video-card";
  button.type = "button";
  button.classList.toggle("is-selected", video.video_id === state.selectedVideoId);
  button.addEventListener("click", () => {
    state.selectedVideoId = video.video_id;
    state.formWarning = "";
    renderVideoLibrary();
    renderSelectedPreview();
    renderFormState();
  });

  const thumb = createThumb(video, false);
  const body = document.createElement("div");
  body.className = "video-card-body";

  const title = document.createElement("p");
  title.className = "video-title";
  title.textContent = getVideoTitle(video);

  const id = document.createElement("div");
  id.className = "video-id";
  id.textContent = video.video_id;

  body.append(title, id);
  button.append(thumb, body);
  return button;
}

function createResultCard(result, rank) {
  const card = document.createElement("article");
  card.className = "result-card";
  card.classList.toggle("is-low", result.low_confidence === true);

  const thumb = createThumb(result, true);

  const rankBadge = document.createElement("span");
  rankBadge.className = "result-rank";
  rankBadge.textContent = `#${rank}`;
  thumb.append(rankBadge);

  const body = document.createElement("div");
  body.className = "result-card-body";

  const title = document.createElement("p");
  title.className = "result-title";
  title.textContent = getVideoTitle(result);

  const id = document.createElement("div");
  id.className = "result-id";
  id.textContent = result.video_id;

  const scoreRow = document.createElement("div");
  scoreRow.className = "score-row";

  const scoreBar = document.createElement("div");
  scoreBar.className = "score-bar";
  const scoreFill = document.createElement("span");
  scoreFill.style.width = `${scoreToPercent(result.score)}%`;
  scoreBar.append(scoreFill);

  const score = document.createElement("span");
  score.className = "score-value";
  score.textContent = `${scoreToPercent(result.score)}%`;

  scoreRow.append(scoreBar, score);
  body.append(title, id, scoreRow);

  if (result.low_confidence) {
    const label = document.createElement("span");
    label.className = "low-label";
    label.textContent = "低置信";
    body.append(label);
  }

  card.append(thumb, body);

  if (state.debug && result.debug) {
    card.append(createDebugPanel(result));
  }

  return card;
}

function createDebugPanel(result) {
  const details = document.createElement("details");
  details.className = "debug-panel";

  const summary = document.createElement("summary");
  summary.textContent = "模型调试信息";

  const content = document.createElement("div");
  content.className = "debug-content";

  const branchScores = result.debug?.branch_scores || {};
  for (const key of ["retain", "inject", "exclude"]) {
    content.append(createBranchLine(key, Number(branchScores[key] ?? 0)));
  }

  if (result.debug?.predicted_action_class) {
    const tag = document.createElement("span");
    tag.className = "action-tag";
    tag.textContent = `动作：${result.debug.predicted_action_class}`;
    content.append(tag);
  }

  const raw = document.createElement("div");
  raw.className = "raw-score";
  raw.textContent = `raw score: ${formatScore(result.score)}`;
  content.append(raw);

  details.append(summary, content);
  return details;
}

function createBranchLine(label, value) {
  const line = document.createElement("div");
  line.className = "debug-line";

  const name = document.createElement("span");
  name.textContent = label;

  const bar = document.createElement("div");
  bar.className = "branch-bar";
  const fill = document.createElement("span");
  fill.style.width = `${scoreToPercent(value)}%`;
  bar.append(fill);

  const score = document.createElement("span");
  score.textContent = formatScore(value);

  line.append(name, bar, score);
  return line;
}

function createThumb(item, previewable) {
  const holder = document.createElement(previewable ? "button" : "div");
  holder.className = previewable ? "thumb thumb-button" : "thumb";
  if (previewable) {
    holder.type = "button";
    holder.addEventListener("click", () => openPreview(item));
  }

  if (item.thumbnail_url) {
    const img = createImage(item.thumbnail_url, getVideoTitle(item));
    holder.append(img);
    attachHoverLoop(holder, img, item);
  } else {
    holder.append(createThumbPlaceholder(item.video_id || "No Preview"));
  }

  return holder;
}

function createImage(src, alt) {
  const img = document.createElement("img");
  img.src = resolveAssetUrl(src);
  img.alt = alt;
  img.loading = "lazy";
  img.addEventListener(
    "error",
    () => {
      const parent = img.parentElement;
      img.remove();
      if (parent && !parent.querySelector(".thumb-placeholder")) {
        parent.prepend(createThumbPlaceholder("预览不可用"));
      }
    },
    { once: true },
  );
  return img;
}

function createThumbPlaceholder(text) {
  const fallback = document.createElement("div");
  fallback.className = "thumb-placeholder";
  fallback.textContent = text;
  return fallback;
}

function createVideoSkeleton() {
  const skeleton = document.createElement("article");
  skeleton.className = "video-card skeleton-card";
  const thumb = document.createElement("div");
  thumb.className = "thumb skeleton-block";
  const line1 = document.createElement("div");
  line1.className = "skeleton-line wide";
  const line2 = document.createElement("div");
  line2.className = "skeleton-line";
  skeleton.append(thumb, line1, line2);
  return skeleton;
}

// Frame-animation plumbing: the backend serves one frame per video by default but
// exposes the full set via /api/frames/<id>; here we loop those frames (~8 fps) so a
// preview "plays" like a short silent clip. previewToken invalidates an in-flight
// loader when a newer preview opens or the dialog closes (async fetch/preload races).
const FRAME_FPS = 8;
const framesCache = new Map(); // video_id -> Promise<string[]> (resolved frame URLs)
let previewTimer = null;
let previewToken = 0;

function stopPreviewLoop() {
  if (previewTimer !== null) {
    window.clearInterval(previewTimer);
    previewTimer = null;
  }
}

function preloadImage(src) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => resolve(true);
    im.onerror = () => resolve(false);
    im.src = src;
  });
}

function loadFrameUrls(videoId) {
  if (framesCache.has(videoId)) return framesCache.get(videoId);
  const promise = api
    .frames(videoId)
    .then((data) => (Array.isArray(data.frames) ? data.frames.map(resolveAssetUrl) : []))
    .catch(() => []);
  framesCache.set(videoId, promise);
  return promise;
}

// Cycle `img.src` through the video's frames. `isCurrent` lets the caller abort if
// the context changed (dialog closed, mouse left) during the async fetch/preload.
async function animateFrames(img, videoId, intervalMs, isCurrent, setTimer) {
  if (USE_MOCK || !videoId) return; // mock data has no real frame folders to fetch
  const urls = await loadFrameUrls(videoId);
  if (!isCurrent() || urls.length < 2) return;

  const loaded = await Promise.all(urls.map(preloadImage));
  const good = urls.filter((_, index) => loaded[index]);
  if (!isCurrent() || good.length < 2) return;

  let index = 0;
  const timer = window.setInterval(() => {
    index = (index + 1) % good.length;
    img.src = good[index];
  }, intervalMs);
  setTimer(timer);
}

// Tiles play on hover. A short hover-intent delay avoids firing requests for tiles
// the pointer merely sweeps across; leaving restores the static poster frame.
function attachHoverLoop(holder, img, item) {
  if (USE_MOCK || !item?.video_id) return;
  const poster = img.src;
  let hoverTimer = null;
  let loopTimer = null;
  let hovering = false;

  const stop = () => {
    hovering = false;
    if (hoverTimer !== null) {
      window.clearTimeout(hoverTimer);
      hoverTimer = null;
    }
    if (loopTimer !== null) {
      window.clearInterval(loopTimer);
      loopTimer = null;
    }
    if (img.src !== poster) img.src = poster;
  };

  holder.addEventListener("mouseenter", () => {
    hovering = true;
    hoverTimer = window.setTimeout(() => {
      animateFrames(img, item.video_id, Math.round(1000 / FRAME_FPS), () => hovering, (t) => {
        if (loopTimer !== null) window.clearInterval(loopTimer);
        loopTimer = t;
      });
    }, 180);
  });
  holder.addEventListener("mouseleave", stop);
}

function openPreview(item) {
  stopPreviewLoop();
  const token = ++previewToken;
  els.dialogMedia.replaceChildren();
  els.dialogId.textContent = item.video_id || "";
  els.dialogTitle.textContent = getVideoTitle(item);

  if (item.video_url) {
    const video = document.createElement("video");
    video.src = resolveAssetUrl(item.video_url);
    video.controls = true;
    video.poster = item.thumbnail_url ? resolveAssetUrl(item.thumbnail_url) : "";
    els.dialogMedia.append(video);
  } else if (item.thumbnail_url) {
    const img = createImage(item.thumbnail_url, getVideoTitle(item));
    els.dialogMedia.append(img);
    animateFrames(img, item.video_id, Math.round(1000 / FRAME_FPS), () => token === previewToken, (t) => {
      if (previewTimer !== null) window.clearInterval(previewTimer);
      previewTimer = t;
    });
  } else {
    els.dialogMedia.append(createThumbPlaceholder("预览不可用"));
  }

  if (typeof els.previewDialog.showModal === "function") {
    els.previewDialog.showModal();
  } else {
    els.previewDialog.setAttribute("open", "");
  }
}

function closePreview() {
  stopPreviewLoop();
  previewToken += 1; // abort any in-flight frame loader for this preview

  const video = els.dialogMedia.querySelector("video");
  if (video) {
    video.pause();
  }

  if (typeof els.previewDialog.close === "function") {
    els.previewDialog.close();
  } else {
    els.previewDialog.removeAttribute("open");
  }
}

function getSelectedVideo() {
  return state.videos.find((video) => video.video_id === state.selectedVideoId) || null;
}

function getSearchInvalidReason() {
  if (!state.selectedVideoId) return "请选择一个参考视频。";
  if (!state.modificationText.trim()) return "请输入修改文本。";
  if (state.topK < 1 || state.topK > MAX_TOP_K) return "返回数量需要在 1 到 50 之间。";
  return "";
}

function getVideoTitle(video) {
  return video.title || video.video_id || "未命名视频";
}

function normalizeOptionalText(value) {
  const text = value.trim();
  return text.length > 0 ? text : null;
}

function scoreToPercent(value) {
  return Math.round(clampNumber(Number(value), 0, 1, 0) * 100);
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "--";
}

function clampNumber(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function getErrorMessage(error, fallback) {
  if (error instanceof ApiError && error.message) return error.message;
  if (error instanceof Error && error.message) return `${fallback}：${error.message}`;
  return fallback;
}

function apiUrl(path, params = {}) {
  const base = API_BASE.replace(/\/$/, "");
  const url = new URL(`${base}${path}`, window.location.origin);

  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }

  return url.toString();
}

function resolveAssetUrl(value) {
  if (!value) return "";
  if (/^(https?:|data:|blob:)/i.test(value)) return value;

  const base = API_BASE
    ? new URL(API_BASE, window.location.origin)
    : new URL(window.location.origin);
  return new URL(value, base).toString();
}

const api = {
  async listVideos({ cursor, limit }) {
    if (USE_MOCK) return mockListVideos(cursor, limit);
    return requestJson(apiUrl("/api/videos", { cursor, limit }));
  },

  async search(payload) {
    if (USE_MOCK) return mockSearch(payload);
    return requestJson(apiUrl("/api/search"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  },

  async frames(videoId, n = 16) {
    return requestJson(apiUrl(`/api/frames/${encodeURIComponent(videoId)}`, { n }));
  },
};

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const message = data?.error?.message || data?.message || response.statusText;
    const code = data?.error?.code || "HTTP_ERROR";
    throw new ApiError(message, code, response.status);
  }

  return data || {};
}

async function mockListVideos(cursor, limit = PAGE_LIMIT) {
  await delay(280);
  const offset = Number(cursor || 0);
  const page = mockVideos.slice(offset, offset + limit);
  const nextOffset = offset + page.length;
  return {
    videos: page,
    next_cursor: nextOffset < mockVideos.length ? String(nextOffset) : null,
  };
}

async function mockSearch(payload) {
  await delay(430);

  const selectedIndex = mockVideos.findIndex(
    (video) => video.video_id === payload.reference_video_id,
  );

  if (selectedIndex === -1) {
    throw new ApiError("reference video not found", "NOT_FOUND", 404);
  }

  const lowOnly = /低置信|不存在|完全不匹配|火山|宇宙/.test(payload.modification_text);
  const topK = clampNumber(Number(payload.top_k), 1, MAX_TOP_K, 20);
  const candidates = mockVideos.filter((video) => video.video_id !== payload.reference_video_id);
  const rotated = candidates
    .map((video, index) => ({
      video,
      distance: Math.abs(index - selectedIndex),
    }))
    .sort((a, b) => a.distance - b.distance);

  const results = rotated.slice(0, topK).map(({ video }, index) => {
    const scoreBase = lowOnly ? 0.28 : 0.91;
    const score = Math.max(0.08, scoreBase - index * (lowOnly ? 0.014 : 0.032));
    return {
      ...video,
      score: Number(score.toFixed(3)),
      low_confidence: score < 0.3 || (index > 8 && !lowOnly),
      debug: payload.debug
        ? {
            branch_scores: {
              retain: Number((0.74 - index * 0.012).toFixed(3)),
              inject: Number((0.66 - index * 0.009).toFixed(3)),
              exclude: Number((0.16 + (index % 4) * 0.035).toFixed(3)),
            },
            predicted_action_class: inferMockAction(payload.modification_text),
          }
        : undefined,
    };
  });

  return {
    query_id: `q_mock_${Date.now().toString(16).slice(-5)}`,
    results,
  };
}

function createMockVideos(count) {
  const titles = [
    "室外跑步的单人镜头",
    "行人穿过城市街口",
    "厨房中打开柜门",
    "室内关闭窗户动作",
    "篮球场上运球移动",
    "骑行经过树荫道路",
    "会议室里挥手示意",
    "草地上慢走回头",
    "桌面物体被拿起",
    "楼梯上向下行走",
  ];

  return Array.from({ length: count }, (_, index) => {
    const id = `vid_${String(index + 1).padStart(6, "0")}`;
    const title = titles[index % titles.length];
    return {
      video_id: id,
      title,
      thumbnail_url: createMockThumbnail(index, title),
      duration_sec: Number((6.4 + (index % 9) * 1.7).toFixed(1)),
    };
  });
}

function createMockThumbnail(index, title) {
  const palettes = [
    ["#0f8b74", "#d96753"],
    ["#425c5a", "#e1a449"],
    ["#6d6f3f", "#1f8a70"],
    ["#884e43", "#2f7567"],
    ["#394b59", "#c9863a"],
  ];
  const [a, b] = palettes[index % palettes.length];
  const label = title.slice(0, 7);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="480" height="270" viewBox="0 0 480 270">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="${a}"/>
          <stop offset="1" stop-color="${b}"/>
        </linearGradient>
      </defs>
      <rect width="480" height="270" fill="url(#g)"/>
      <rect x="32" y="34" width="416" height="202" rx="18" fill="rgba(255,255,255,0.13)" stroke="rgba(255,255,255,0.2)"/>
      <circle cx="${118 + (index % 5) * 52}" cy="108" r="26" fill="rgba(255,255,255,0.42)"/>
      <path d="M84 210 C150 142, 226 226, 314 146 S422 178, 448 130" fill="none" stroke="rgba(255,255,255,0.55)" stroke-width="10" stroke-linecap="round"/>
      <text x="38" y="64" fill="rgba(255,255,255,0.88)" font-family="Arial, sans-serif" font-size="26" font-weight="700">${label}</text>
      <text x="38" y="232" fill="rgba(255,255,255,0.76)" font-family="Arial, sans-serif" font-size="18">${String(index + 1).padStart(2, "0")}</text>
    </svg>`;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function inferMockAction(text) {
  if (/走|步行/.test(text)) return "walk";
  if (/跑/.test(text)) return "run";
  if (/打开/.test(text)) return "open";
  if (/关闭|关上/.test(text)) return "close";
  if (/骑/.test(text)) return "ride";
  return "compose";
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

// Start the app only after every top-level declaration above (notably `const api`)
// has initialized — calling init() earlier hits a temporal-dead-zone ReferenceError
// in loadVideos() ("Cannot access 'api' before initialization").
init();
