'use strict';

const $ = (s) => document.querySelector(s);
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  const t = await r.text();
  try { return JSON.parse(t); } catch { return { error: t }; }
};
const post = (url, body) => api(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
  body: JSON.stringify(body || {}),
});
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const size = (n) => n > 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.round(n / 1024) + ' KB';

const LANGS = [
  ['auto', '自动检测'], ['en', '英语'], ['ja', '日语'], ['ko', '韩语'],
  ['fr', '法语'], ['de', '德语'], ['es', '西班牙语'], ['ru', '俄语'],
  ['pt', '葡萄牙语'], ['it', '意大利语'], ['zh', '中文'],
];

const S = { files: [], job: null, src: null, state: null, termFile: null, terms: [], gloss: [] };

/* ---------------- 视图切换 ---------------- */
document.querySelectorAll('.nav').forEach((el) => {
  el.onclick = () => {
    document.querySelectorAll('.nav').forEach((n) => n.classList.toggle('on', n === el));
    document.querySelectorAll('.view').forEach((v) => v.classList.remove('on'));
    $('#v-' + el.dataset.view).classList.add('on');
    if (el.dataset.view === 'jobs') loadJobs();
    if (el.dataset.view === 'terms') loadTermList();
  };
});

const banner = (msg, kind) => {
  const b = $('#banner');
  b.className = 'banner on ' + (kind || 'warn');
  b.innerHTML = msg;
};
const hideBanner = () => $('#banner').classList.remove('on');

/* ---------------- 术语表选择 ---------------- */
async function loadGlossPicker(selected) {
  const r = await api('/api/terms');
  S.gloss = r.files || [];
  const names = selected && selected.length ? selected : null;
  $('#glossPicker').innerHTML = S.gloss.map((f) => {
    const on = !names || names.includes(f.name);
    return `<label class="chip${on ? ' on' : ''}">
      <input type="checkbox" data-g="${esc(f.name)}"${on ? ' checked' : ''}>
      ${esc(f.name.replace(/-zh\.csv$/, ''))}<em>${f.entries}</em></label>`;
  }).join('') || '<div class="empty">glossary/ 目录下还没有术语表</div>';
  syncGloss();
  $('#glossPicker').querySelectorAll('input').forEach((i) => (i.onchange = () => {
    i.closest('.chip').classList.toggle('on', i.checked);
    syncGloss();
  }));
}
const glossNames = () =>
  [...$('#glossPicker').querySelectorAll('input:checked')].map((i) => i.dataset.g);
const syncGloss = () => {
  const n = glossNames().length;
  const total = S.gloss.reduce((a, f) => (glossNames().includes(f.name) ? a + f.entries : a), 0);
  $('#glossCount').textContent = n ? `${n} 表 / ${total} 条` : '未启用';
};
$('#glossAll').onclick = () => {
  $('#glossPicker').querySelectorAll('input').forEach((i) => {
    i.checked = true; i.closest('.chip').classList.add('on');
  });
  syncGloss();
};
$('#glossNone').onclick = () => {
  $('#glossPicker').querySelectorAll('input').forEach((i) => {
    i.checked = false; i.closest('.chip').classList.remove('on');
  });
  syncGloss();
};

/* ---------------- 初始化 ---------------- */
async function boot() {
  LANGS.forEach(([v, t]) => {
    $('#langIn').insertAdjacentHTML('beforeend', `<option value="${v}">${t}</option>`);
    if (v !== 'auto') $('#langOut').insertAdjacentHTML('beforeend', `<option value="${v}">${t}</option>`);
  });
  $('#langIn').value = 'auto';
  $('#langOut').value = 'zh';

  const st = await api('/api/state');
  S.state = st;

  const p = st.params || {};
  if (p.model) $('#model').value = p.model;
  if (p.base_url) $('#baseUrl').value = p.base_url;
  if (p.qps) $('#qps').value = p.qps;
  if (p.lang_out) $('#langOut').value = p.lang_out;
  if (p.font_scale) { $('#fontScale').value = p.font_scale; $('#fsVal').textContent = (+p.font_scale).toFixed(2); }
  if (p.pages) $('#pages').value = p.pages;
  if (p.skip_scanned === false) $('#skipScanned').checked = false;
  if (p.rebuild_scrambled === false) $('#rebuildScrambled').checked = false;
  if (p.auto_fill === false) $('#autoFill').checked = false;
  if (p.ocr_workaround === true) $('#ocrWork').checked = true;
  if (p.no_terms === true) $('#noTerms').checked = true;
  if (p.dual_only === true) $('#dualOnly').checked = true;
  if (p.mono_only === true) $('#monoOnly').checked = true;

  $('#dotEngine').classList.toggle('on', !!st.engine_ok);
  $('#txtEngine').textContent = st.engine_ok ? '引擎就绪' : '引擎缺失';
  $('#dotKey').classList.toggle('on', !!st.api_key_set);
  $('#txtKey').textContent = st.api_key_set ? '密钥已设置' : '密钥未设置';

  $('#envRows').innerHTML = [
    ['引擎', st.engine || '—'],
    ['工作目录', st.root],
    ['输出目录', st.out_dir],
    ['可执行文件', st.engine_ok ? 'BabelDOC 0.6.4' : '未找到'],
  ].map(([k, v]) => `<tr><th style="width:150px">${k}</th><td style="word-break:break-all">${esc(v)}</td></tr>`).join('');

  await loadGlossPicker(p.glossary_files);

  if (!st.api_key_set) banner('还没有配置 DeepSeek API Key，先到「设置」里填一个再翻译。', 'warn');
}

/* ---------------- 文件 ---------------- */
function renderFiles() {
  const box = $('#files');
  if (!S.files.length) { box.innerHTML = ''; return; }
  box.innerHTML = S.files.map((f, i) => `
    <div class="file">
      <span class="nm">${esc(f.name)}</span>
      <span class="sz">${f.size ? size(f.size) : ''}</span>
      <span class="x" data-i="${i}">✕</span>
    </div>`).join('');
  box.querySelectorAll('.x').forEach((x) => {
    x.onclick = () => { S.files.splice(+x.dataset.i, 1); renderFiles(); };
  });
}
const addFiles = (f) => {
  f.forEach((x) => { if (!S.files.some((y) => y.path === x.path)) S.files.push(x); });
  renderFiles();
};

$('#drop').onclick = () => ($('#pick').click());
$('#pick').onchange = async (e) => {
  for (const f of e.target.files) {
    const r = await fetch('/api/upload?name=' + encodeURIComponent(f.name), { method: 'POST', body: f });
    const j = await r.json();
    if (j.path) addFiles([{ name: f.name, path: j.path, size: f.size }]);
    else banner('上传失败：' + esc(j.error || '未知错误'), 'err');
  }
  e.target.value = '';
};
['dragenter', 'dragover'].forEach((ev) => $('#drop').addEventListener(ev, (e) => {
  e.preventDefault(); $('#drop').classList.add('over');
}));
['dragleave', 'drop'].forEach((ev) => $('#drop').addEventListener(ev, (e) => {
  e.preventDefault(); $('#drop').classList.remove('over');
}));
$('#drop').addEventListener('drop', async (e) => {
  const fs = [...(e.dataTransfer?.files || [])].filter((f) => /\.pdf$/i.test(f.name));
  if (!fs.length) return;
  for (const f of fs) {
    const r = await fetch('/api/upload?name=' + encodeURIComponent(f.name), { method: 'POST', body: f });
    const j = await r.json();
    if (j.path) addFiles([{ name: f.name, path: j.path, size: f.size }]);
  }
});

/* 浏览弹窗 */
async function browse(dir) {
  const r = await api('/api/browse?dir=' + encodeURIComponent(dir || ''));
  if (r.error) { banner('无法读取目录：' + esc(r.error), 'err'); return; }
  $('#mPath').textContent = r.dir + '   （当前目录内 PDF 与子目录）';
  let html = '';
  if (r.parent) html += `<div class="brow" data-go="${esc(r.parent)}">📁 <b>..</b></div>`;
  html += (r.dirs || []).map((d) => `<div class="brow" data-go="${esc(d.path)}">📁 ${esc(d.name)}</div>`).join('');
  html += (r.pdfs || []).map((f) => `<div class="brow" data-pdf="${esc(f.path)}" data-name="${esc(f.name)}" data-size="${f.size}">📄 ${esc(f.name)}<span class="sz">${size(f.size)}</span></div>`).join('');
  if (!html) html = '<div class="empty">此目录没有 PDF</div>';
  $('#mBody').innerHTML = html;
  $('#mBody').querySelectorAll('[data-go]').forEach((el) => (el.onclick = () => browse(el.dataset.go)));
  $('#mBody').querySelectorAll('[data-pdf]').forEach((el) => (el.onclick = () => {
    addFiles([{ name: el.dataset.name, path: el.dataset.pdf, size: +el.dataset.size }]);
    $('#modal').classList.remove('on');
  }));
  $('#modal').classList.add('on');
}
$('#mClose').onclick = () => $('#modal').classList.remove('on');
$('#modal').onclick = (e) => { if (e.target.id === 'modal') $('#modal').classList.remove('on'); };

/* ---------------- 翻译 ---------------- */
const params = () => ({
  lang_in: $('#langIn').value,
  lang_out: $('#langOut').value,
  model: $('#model').value,
  base_url: $('#baseUrl').value || 'https://api.deepseek.com/v1',
  qps: +$('#qps').value || 4,
  pages: $('#pages').value.trim(),
  font_scale: +$('#fontScale').value,
  // With auto-selection on, send no explicit list so the backend scores the
  // document and loads only the tables it actually needs.
  builtin_glossary: $('#autoGloss').checked || glossNames().length > 0,
  glossary_files: $('#autoGloss').checked ? [] : glossNames(),
  auto_fill: $('#autoFill').checked,
  bilingual_view: $('#bilingual').checked,
  auto_glossary: $('#autoGloss').checked,
  no_terms: $('#noTerms').checked,
  skip_scanned: $('#skipScanned').checked,
  rebuild_scrambled: $('#rebuildScrambled').checked,
  ocr_workaround: $('#ocrWork').checked,
  dual_only: $('#dualOnly').checked,
  mono_only: $('#monoOnly').checked,
  watermark: false,
});

$('#fontScale').oninput = (e) => ($('#fsVal').textContent = (+e.target.value).toFixed(2));

$('#btnRun').onclick = async () => {
  if (!S.files.length) { banner('先选一个 PDF。', 'warn'); return; }
  hideBanner();
  $('#prog').style.display = 'block';
  $('#log').innerHTML = '';
  $('#bar').style.width = '0%';
  $('#pct').textContent = '0%';
  $('#stage').textContent = '提交中…';
  $('#outs').innerHTML = '<div class="empty">翻译进行中…</div>';
  $('#btnRun').disabled = true;
  $('#btnCancel').disabled = false;

  const r = await post('/api/run', { files: S.files.map((f) => f.path), params: params() });
  if (r.error || !r.job) {
    $('#btnRun').disabled = false; $('#btnCancel').disabled = true;
    banner('提交失败：' + esc(r.error || '未知错误'), 'err');
    return;
  }
  S.job = r.job;
  stream(r.job);
};

$('#btnCancel').onclick = async () => {
  if (!S.job) return;
  await post('/api/cancel', { job: S.job });
  $('#stage').textContent = '正在取消…';
};

function stream(job) {
  const es = new EventSource('/api/stream?job=' + encodeURIComponent(job));
  es.onmessage = (ev) => {
    let d; try { d = JSON.parse(ev.data); } catch { return; }
    if (d.type === 'progress') {
      const v = Math.max(0, Math.min(100, +d.progress || 0));
      $('#bar').style.width = v + '%';
      $('#pct').textContent = v.toFixed(1) + '%';
      if (d.stage) $('#stage').textContent = d.stage;
      // One engine stage can take a minute; the running clock is what tells the
      // user the job is alive rather than stuck.
      if (d.elapsed != null) {
        const s = Math.round(d.elapsed);
        $('#elapsed').textContent = s < 60
          ? '已用时 ' + s + ' 秒'
          : '已用时 ' + Math.floor(s / 60) + ' 分 ' + (s % 60) + ' 秒';
      }
    } else if (d.type === 'log' || d.type === 'cmd') {
      const cls = d.type === 'cmd' ? 'l-cmd' : '';
      $('#log').insertAdjacentHTML('beforeend', `<div class="${cls}">${esc(d.text)}</div>`);
      $('#log').scrollTop = $('#log').scrollHeight;
    } else if (d.type === 'warn') {
      $('#log').insertAdjacentHTML('beforeend', `<div class="l-warn">⚠ ${esc(d.text)}</div>`);
      $('#log').scrollTop = $('#log').scrollHeight;
      banner(esc(d.text), 'warn');
    } else if (d.type === 'error') {
      $('#log').insertAdjacentHTML('beforeend', `<div class="l-err">✕ ${esc(d.text)}</div>`);
      $('#log').scrollTop = $('#log').scrollHeight;
      banner(esc(d.text), 'err');
    } else if (d.type === 'end') {
      es.close();
      $('#btnRun').disabled = false;
      $('#btnCancel').disabled = true;
      $('#bar').style.width = '100%';
      $('#pct').textContent = '100%';
      $('#stage').textContent = d.rc === 0 ? '完成' : '结束（退出码 ' + d.rc + '）';
      $('#elapsed').textContent = '';
      renderOutputs(d.outputs || [], d.compare_html);
      loadJobs();
    }
  };
  es.onerror = () => es.close();
}

function renderOutputs(list, compareHtml) {
  const box = $('#outs');
  if (!list.length) { box.innerHTML = '<div class="empty">没有产出文件</div>'; return; }
  const cmp = compareHtml
    ? `<div style="margin:0 0 10px"><button class="sm pri" data-prev="${esc(compareHtml)}">打开逐段对照阅读视图</button></div>`
    : '';
  box.innerHTML = cmp + `<table><tbody>${list.map((p) => {
    const n = p.split(/[\\/]/).pop();
    const tag = /dual/i.test(n) ? '双语对照' : (/mono/i.test(n) ? '纯译文' : '');
    return `<tr>
      <td>${esc(n)} ${tag ? `<span class="pill">${tag}</span>` : ''}</td>
      <td class="act">
        <button class="sm pri" data-open="${esc(p)}">用 Edge 打开</button>
        <button class="sm" data-dl="${esc(p)}">下载</button>
        <button class="sm" data-rev="${esc(p)}">定位</button>
      </td></tr>`;
  }).join('')}</tbody></table>`;
  wireFileButtons(box);
}

function wireFileButtons(scope) {
  scope.querySelectorAll('[data-open]').forEach((b) => (b.onclick = async () => {
    const r = await post('/api/open', { path: b.dataset.open });
    if (r.error) banner('打开失败：' + esc(r.error), 'err');
    else if (r.via === 'shell') banner('已交给系统默认程序打开（未找到 Edge）。', 'warn');
  }));
  scope.querySelectorAll('[data-dl]').forEach((b) => (b.onclick = () =>
    (location.href = '/api/download?path=' + encodeURIComponent(b.dataset.dl))));
  scope.querySelectorAll('[data-prev]').forEach((b) => (b.onclick = () =>
    window.open('/api/preview?path=' + encodeURIComponent(b.dataset.prev), '_blank')));
  scope.querySelectorAll('[data-rev]').forEach((b) => (b.onclick = () =>
    post('/api/reveal', { path: b.dataset.rev })));
}

$('#btnOpenOut').onclick = () => post('/api/reveal', { path: S.state?.out_dir || '' });

/* ---------------- 任务历史 ---------------- */
async function loadJobs() {
  const r = await api('/api/jobs');
  const rows = r.jobs || [];
  if (!rows.length) { $('#jobRows').innerHTML = '<tr><td colspan="6" class="empty">暂无任务</td></tr>'; return; }
  $('#jobRows').innerHTML = rows.map((j) => {
    const names = (j.files || []).map((f) => f.split(/[\\/]/).pop()).join('、');
    const d = new Date((j.started || 0) * 1000);
    return `<tr>
      <td>${isNaN(d) ? '—' : d.toLocaleString('zh-CN', { hour12: false })}</td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(names)}">${esc(names)}</td>
      <td><span class="pill ${esc(j.status)}">${esc(j.status)}</span></td>
      <td>${(+j.progress || 0).toFixed(0)}%</td>
      <td>${(j.prompt_tokens || 0)}/${(j.completion_tokens || 0)}</td>
      <td class="act">${
        (j.compare_html
          ? `<button class="sm primary" data-prev="${esc(j.compare_html)}">对照阅读</button> `
          : '')
      }${(j.outputs || []).map((p) => `<button class="sm" data-open="${esc(p)}">打开</button>`).join(' ')}</td>
    </tr>`;
  }).join('');
  wireFileButtons($('#jobRows'));
}

/* ---------------- 术语库 ---------------- */
async function loadTermList() {
  const r = await api('/api/terms');
  const files = r.files || [];
  if (!files.length) { $('#termList').innerHTML = '<div class="empty">还没有术语表</div>'; return; }
  $('#termList').innerHTML = files.map((f) => `
    <div class="it${S.termFile === f.name ? ' on' : ''}" data-n="${esc(f.name)}">
      ${esc(f.name)}<small>${f.entries} 条</small>
    </div>`).join('');
  $('#termList').querySelectorAll('.it').forEach((el) => (el.onclick = () => openTerm(el.dataset.n)));
}

async function openTerm(name) {
  S.termFile = name;
  const r = await api('/api/terms?name=' + encodeURIComponent(name));
  S.terms = r.entries || [];
  $('#termName').textContent = name;
  $('#termCount').textContent = S.terms.length + ' 条';
  renderTerms();
  loadTermList();
}

function renderTerms() {
  if (!S.terms.length) { $('#termRows').innerHTML = '<div class="empty">空表，点「加一行」开始</div>'; return; }
  $('#termRows').innerHTML = `<table>
    <thead><tr><th style="width:44%">原文（source）</th><th style="width:44%">译文（target）</th><th></th></tr></thead>
    <tbody>${S.terms.map((t, i) => `<tr>
      <td><input data-i="${i}" data-k="source" value="${esc(t.source)}"></td>
      <td><input data-i="${i}" data-k="target" value="${esc(t.target)}"></td>
      <td class="act"><button class="sm danger" data-del="${i}">删</button></td>
    </tr>`).join('')}</tbody></table>`;
  $('#termRows').querySelectorAll('input').forEach((inp) => (inp.oninput = () => {
    S.terms[+inp.dataset.i][inp.dataset.k] = inp.value;
  }));
  $('#termRows').querySelectorAll('[data-del]').forEach((b) => (b.onclick = () => {
    S.terms.splice(+b.dataset.del, 1);
    $('#termCount').textContent = S.terms.length + ' 条';
    renderTerms();
  }));
}

$('#btnAddRow').onclick = () => {
  if (!S.termFile) { banner('先选择或新建一个术语表。', 'warn'); return; }
  S.terms.push({ source: '', target: '', tgt_lng: 'zh' });
  $('#termCount').textContent = S.terms.length + ' 条';
  renderTerms();
};

$('#btnSaveTerm').onclick = async () => {
  if (!S.termFile) { banner('还没有打开的术语表。', 'warn'); return; }
  const clean = S.terms.filter((t) => String(t.source).trim() && String(t.target).trim());
  const r = await post('/api/terms', { name: S.termFile, entries: clean });
  if (r.error) banner('保存失败：' + esc(r.error), 'err');
  else { banner(`已保存 ${r.entries} 条到 ${esc(r.name)}。`, 'ok'); S.terms = clean; renderTerms(); loadTermList(); }
};

$('#btnNewTerm').onclick = async () => {
  const n = prompt('新术语表文件名：', 'my-terms-zh.csv');
  if (!n) return;
  const name = /\.csv$/i.test(n) ? n : n + '.csv';
  const r = await post('/api/terms', { name, entries: [] });
  if (r.error) banner('创建失败：' + esc(r.error), 'err');
  else { await loadTermList(); openTerm(r.name); }
};

/* ---------------- 设置 ---------------- */
$('#btnSaveKey').onclick = async () => {
  const r = await post('/api/config', { api_key: $('#apiKey').value.trim(), params: params() });
  if (r.error) banner('保存失败：' + esc(r.error), 'err');
  else {
    banner('设置已保存。', 'ok');
    const st = await api('/api/state');
    S.state = st;
    $('#dotKey').classList.toggle('on', !!st.api_key_set);
    $('#txtKey').textContent = st.api_key_set ? '密钥已设置' : '密钥未设置';
  }
};
$('#btnShowKey').onclick = () => {
  const el = $('#apiKey');
  el.type = el.type === 'password' ? 'text' : 'password';
};

boot();
