// frontend/app.js

// Ensure BACKEND_URL is available from config.js, or use a default
const API_BASE = typeof BACKEND_URL !== 'undefined' ? BACKEND_URL : "https://llm-gateway-62xo.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
    // 1. Authentication Check
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/index.html';
        return;
    }

    // 2. DOM Elements
    const form = document.getElementById('composer');
    const queryInput = document.getElementById('query');
    const sendBtn = document.getElementById('send');
    const thread = document.getElementById('thread');
    const healthIndicator = document.getElementById('health');
    const modeSelect = document.getElementById('mode');
    const explainCheckbox = document.getElementById('explain');

    // 3. Health Check
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            const data = await res.json();
            if (data.status === 'ok') {
                healthIndicator.className = 'health ok';
                healthIndicator.title = "Gateway / Router Status: OK";
            } else {
                healthIndicator.className = 'health degraded';
                healthIndicator.title = "Gateway / Router Status: Degraded";
            }
        } catch (e) {
            healthIndicator.className = 'health down';
            healthIndicator.title = "Gateway / Router Status: Down";
        }
    }
    checkHealth();

    // 4. Input Handling (Shift+Enter for newline, Enter to send)
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) {
                form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            }
        }
    });

    // 5. Submit Message to Backend
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (!text) return;

        const currentMode = modeSelect.value;
        const wantsExplanation = explainCheckbox.checked;

        // Update UI
        appendUserMessage(text);
        queryInput.value = '';
        sendBtn.disabled = true;
        queryInput.disabled = true;

        try {
            const response = await fetch(`${API_BASE}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    query: text,
                    mode: currentMode,
                    explain: wantsExplanation,
                    use_cache: true
                })
            });

            // Handle unauthorized access (token expired or invalid)
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/index.html';
                return;
            }

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Server error occurred');
            }

            const data = await response.json();
            appendAssistantMessage(data, currentMode);

        } catch (err) {
            console.error("Chat error:", err);
            appendErrorMessage(err.message);
        } finally {
            sendBtn.disabled = false;
            queryInput.disabled = false;
            queryInput.focus();
        }
    });

    // 6. UI Rendering Functions
    function appendUserMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg user';
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text; // textContent prevents XSS
        
        msgDiv.appendChild(bubble);
        thread.appendChild(msgDiv);
        thread.scrollTop = thread.scrollHeight;
    }

    function appendAssistantMessage(data, currentMode) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg assistant';

        // Extract backend keys based on gateway/app.py ChatResponse structure
        const decision = data.decision || {};
        const routeDecision = decision.tier || 'LOW';
        const pStrong = decision.p_strong ? decision.p_strong.toFixed(3) : '0.000';
        const latency = data.router_latency_ms || 0;
        const reasoning = decision.reasoning || '';
        const answerText = data.answer || '';

        // Build Metadata Header
        const badgeClass = routeDecision.toLowerCase() === 'high' ? 'high' : 'low';
        const metaHTML = `
            <div class="meta">
                <span class="tier-badge ${badgeClass}">${routeDecision.toUpperCase()}</span>
                <span>p(strong)=${pStrong} mode=${currentMode} ${latency} ms</span>
            </div>
        `;
        msgDiv.insertAdjacentHTML('beforeend', metaHTML);

        // Build Message Bubble
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bubble markdown';
        
        if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
            // Configure marked for GitHub Flavored Markdown
            marked.setOptions({
                gfm: true,
                breaks: true
            });
            bubbleDiv.innerHTML = DOMPurify.sanitize(marked.parse(answerText));
        } else {
            bubbleDiv.innerText = answerText;
        }
        msgDiv.appendChild(bubbleDiv);

        // Build Routing Explanation
        if (explainCheckbox.checked && reasoning) {
            const reasoningDiv = document.createElement('div');
            reasoningDiv.className = 'reasoning';
            reasoningDiv.innerHTML = reasoning.replace(/\n/g, '<br>');
            msgDiv.appendChild(reasoningDiv);
        }

        thread.appendChild(msgDiv);
        
        // Trigger syntax highlighting
        if (typeof hljs !== 'undefined') {
            msgDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }

        thread.scrollTop = thread.scrollHeight;
    }

    function appendErrorMessage(errorMsg) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg assistant';
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble error-bubble';
        bubble.style.borderColor = 'var(--error)';
        bubble.style.color = 'var(--error)';
        bubble.textContent = 'Error: ' + errorMsg;
        
        msgDiv.appendChild(bubble);
        thread.appendChild(msgDiv);
        thread.scrollTop = thread.scrollHeight;
    }

    // --- Phase 2C: shared authed-fetch helper (same 401 -> redirect behavior
    // as the main chat request above, reused for the usage/admin panels) ---
    async function authedFetch(path, opts = {}) {
        const res = await fetch(`${API_BASE}${path}`, {
            ...opts,
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}`, ...(opts.headers || {}) },
        });
        if (res.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/index.html';
            throw new Error('Session expired');
        }
        return res;
    }

    // --- Phase 2C: "My Usage" panel (GET /api/me/usage) ---
    const usageBtn = document.getElementById('usageBtn');
    const usagePanel = document.getElementById('usagePanel');
    const usageContent = document.getElementById('usageContent');
    const adminBtn = document.getElementById('adminBtn');
    const adminPanel = document.getElementById('adminPanel');
    const adminContent = document.getElementById('adminContent');

    function togglePanel(panel) { panel.classList.toggle('hidden'); }
    document.querySelectorAll('.panel-close').forEach((btn) => {
        btn.addEventListener('click', () => document.getElementById(btn.dataset.close).classList.add('hidden'));
    });

    function renderUsageBar(used, limit, label) {
        if (limit == null) {
            return `<div class="usage-row"><span>${label}</span><span class="usage-unlimited">unlimited</span></div>`;
        }
        const pct = Math.min(100, Math.round((used / limit) * 100));
        const barClass = pct >= 100 ? 'danger' : pct >= 75 ? 'warn' : 'ok';
        return `
            <div class="usage-row"><span>${label}</span><span>${used} / ${limit}</span></div>
            <div class="usage-bar"><div class="usage-bar-fill ${barClass}" style="width:${pct}%"></div></div>
        `;
    }

    async function loadUsage() {
        usageContent.innerHTML = 'Loading…';
        try {
            const res = await authedFetch('/api/me/usage');
            if (!res.ok) { usageContent.innerHTML = 'No policy assigned yet.'; return; }
            const d = await res.json();
            usageContent.innerHTML = `
                <p class="usage-meta">Policy: <strong>${d.policy_name}</strong> ·
                    action on exhaustion: <code>${d.action_on_exhaustion}</code> · resets monthly</p>
                ${renderUsageBar(d.queries_used, d.query_threshold, 'Queries this month')}
                ${d.token_threshold != null ? renderUsageBar(d.tokens_used, d.token_threshold, 'Tokens this month') : ''}
            `;
        } catch (e) {
            usageContent.innerHTML = `<span class="error-text">Could not load usage: ${e.message}</span>`;
        }
    }

    usageBtn.addEventListener('click', () => {
        togglePanel(usagePanel);
        if (!usagePanel.classList.contains('hidden')) loadUsage();
    });

    // --- Phase 2C: Admin panel (GET /api/admin/users + /api/admin/policies,
    // PATCH /api/admin/users/{id}/policy to reassign) ---
    async function loadAdmin() {
        adminContent.innerHTML = 'Loading…';
        try {
            const [usersRes, policiesRes] = await Promise.all([
                authedFetch('/api/admin/users'),
                authedFetch('/api/admin/policies'),
            ]);
            if (!usersRes.ok || !policiesRes.ok) { adminContent.innerHTML = 'Admin access required.'; return; }
            const users = await usersRes.json();
            const policies = await policiesRes.json();

            const policyOptions = policies.map((p) => `<option value="${p.id}">${p.name}</option>`).join('');
            adminContent.innerHTML = `
                <table class="admin-table">
                    <thead><tr><th>User</th><th>Usage (this month)</th><th>Policy</th></tr></thead>
                    <tbody>
                        ${users.map((u) => `
                            <tr data-user-id="${u.id}">
                                <td>${u.email}${u.is_admin ? ' <span class="admin-badge">admin</span>' : ''}</td>
                                <td>${u.queries_used} q / ${u.tokens_used} tok</td>
                                <td>
                                    <select class="policy-select">${policyOptions}</select>
                                    <button type="button" class="reassign-btn">Save</button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            users.forEach((u) => {
                if (u.policy_id == null) return;
                const sel = adminContent.querySelector(`tr[data-user-id="${u.id}"] .policy-select`);
                if (sel) sel.value = u.policy_id;
            });
            adminContent.querySelectorAll('.reassign-btn').forEach((btn) => {
                btn.addEventListener('click', async () => {
                    const row = btn.closest('tr');
                    const userId = row.dataset.userId;
                    const policyId = parseInt(row.querySelector('.policy-select').value, 10);
                    btn.disabled = true;
                    btn.textContent = 'Saving…';
                    try {
                        const res = await authedFetch(`/api/admin/users/${userId}/policy`, {
                            method: 'PATCH',
                            body: JSON.stringify({ policy_id: policyId }),
                        });
                        btn.textContent = res.ok ? 'Saved ✓' : 'Failed';
                    } catch (e) {
                        btn.textContent = 'Failed';
                    } finally {
                        setTimeout(() => { btn.disabled = false; btn.textContent = 'Save'; }, 1500);
                    }
                });
            });
        } catch (e) {
            adminContent.innerHTML = `<span class="error-text">Could not load admin data: ${e.message}</span>`;
        }
    }

    adminBtn.addEventListener('click', () => {
        togglePanel(adminPanel);
        if (!adminPanel.classList.contains('hidden')) loadAdmin();
    });

    // Reveal the Admin button only for accounts that actually pass require_admin
    // server-side - this is a UI convenience, not the security boundary (the
    // /api/admin/* endpoints enforce it themselves regardless of this check).
    authedFetch('/api/admin/policies').then((res) => {
        if (res.ok) adminBtn.style.display = '';
    }).catch(() => {});
});
