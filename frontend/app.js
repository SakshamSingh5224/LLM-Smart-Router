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
});
