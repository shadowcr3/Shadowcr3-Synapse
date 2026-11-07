(function (S) {
  'use strict';
  const { $, api } = S.dom;
  const TABS = ['predict', 'train', 'model'];
  const st = { timer: null, readOnly: false };

  function showTab(name) {
    TABS.forEach((t) => {
      const on = t === name;
      $('tab-' + t).hidden = !on;
      const b = $('tabbtn-' + t);
      b.classList.toggle('active', on); b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    if (name === 'model') S.model.refresh();
    if (name === 'train') S.train.refreshFeedback();
    if (history.replaceState) history.replaceState(null, '', '#' + name);
  }

  function setStatus(kind, text) {
    $('status-pill').className = 'status ' + kind;
    $('status-text').textContent = text;
  }

  function applyReadOnly(flag) {
    if (flag === st.readOnly) return;
    st.readOnly = flag;
    ['btn-train', 'btn-sweep', 'btn-ft', 'btn-finetune', 'btn-ft-clear', 'btn-save-sample', 'btn-undo', 'btn-rollback'].forEach((id) => {
      const n = $(id); if (n) n.disabled = flag || n.disabled;
    });
    if (flag) ['t-msg', 's-msg', 'ft-msg', 'fb-status'].forEach((id) => {
      $(id).textContent = 'This server is read-only: training, saving drawings and changing the network are switched off.';
    });
  }

  async function checkHealth() {
    try {
      const h = await api('/health');
      setStatus('ok', 'model loaded, ' + (h.active_model === 'release' ? 'release' : h.active_model) + (h.read_only ? ', read-only' : '') + (h.training_busy ? ', training' : ''));
      applyReadOnly(h.read_only);
      return h;
    } catch (e) {
      if (e.code === 'rate_limited') return null;
      setStatus('err', 'server offline');
      return null;
    }
  }

  async function init() {
    TABS.forEach((t) => $('tabbtn-' + t).addEventListener('click', () => showTab(t)));
    let digits;
    try { digits = (await api('/model/info')).digits; } catch (e) { setStatus('err', 'server offline'); }
    S.predict.init(digits);
    S.train.init();
    S.model.init();
    await checkHealth();
    st.timer = setInterval(checkHealth, 10000);
    const start = (location.hash || '').replace('#', '');
    if (TABS.indexOf(start) > 0) showTab(start);
  }

  S.main = { init, showTab, checkHealth, st };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})(window.Synapse = window.Synapse || {});
