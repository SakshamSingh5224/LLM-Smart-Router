// Configure marked.js to use highlight.js for rendering code blocks
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true // Enables GitHub-flavored markdown line breaks
});

// 1. Authentication Check
const token = localStorage.getItem('token');
// Redirect to the new login page (index.html or /) if unauthenticated
if (!token && window.location.pathname !== '/' && window.location.pathname !== '/index.html') {
    window.location.replace('/');
}

// 2. DOM Elements
const chatForm = document.getElementById('composer');
const chatInput = document.getElementById('query');
const chatContainer = document.getElementById('thread');

// Only run the chat logic if we are actually on the chat page
if (chatForm && chatInput && chatContainer) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Display user message as plain text (false = no markdown parsing)
        appendMessage('user', query, false);
        chatInput.value = '';

        // Create a placeholder for the assistant's streaming response
        const assistantBubble = appendMessage('assistant', '', true);
        
        // Accumulate the full text safely before rendering
        let fullText = ""; 

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ query: query })
            });

            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.replace('/');
                return;
            }

            if (!response.ok) {
                assistantBubble.innerHTML = `<span style="color:red">Error: ${response.statusText}</span>`;
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                
                const parts = buffer.split('\n\n');
                buffer = parts.pop(); 

                for (const part of parts) {
                    let eventType = 'message';
                    let data = null;

                    const lines = part.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            eventType = line.substring(7).trim();
                        } else if (line.startsWith('data: ')) {
                            const dataString = line.substring(6).trim();
                            try {
                                data = JSON.parse(dataString);
                            } catch (parseError) {
                                continue; 
                            }
                        }
                    }

                    if (eventType === 'meta' && data) {
                        console.log("Routed to:", data.decision.tier);
                    } else if (eventType === 'delta' && data && data.text) {
                        fullText += data.text;
                        
                        // Parse Markdown to HTML, then sanitize it safely
                        const rawHTML = marked.parse(fullText);
                        assistantBubble.innerHTML = DOMPurify.sanitize(rawHTML);
                        
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    } else if (eventType === 'error' && data) {
                        assistantBubble.innerHTML += `<br><span style="color:red">Generation Error: ${data.message}</span>`;
                    } else if (eventType === 'done' && data) {
                        console.log(`Stream finished. Latency: ${data.total_latency_ms}ms`);
                    }
                }
            }
        } catch (error) {
            console.error("Fetch error:", error);
            assistantBubble.innerHTML += `<br><span style="color:red">Network Error: ${error.message}</span>`;
        }
    });
}

// Helper function to render messages safely supporting Markdown
function appendMessage(role, text, isMarkdown = false) {
    if (!chatContainer) return null;
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    if (text) {
        if (isMarkdown) {
            msgDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));
        } else {
            msgDiv.textContent = text; 
        }
    }
    
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msgDiv;
}

// Optional: Attach this to a logout button in your HTML
function logout() {
    localStorage.removeItem('token');
    window.location.replace('/');
}
