// 1. Authentication Check
const token = localStorage.getItem('token');
if (!token && window.location.pathname !== '/login.html') {
    window.location.href = '/login.html';
}

// 2. DOM Elements (Safely selected)
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatContainer = document.getElementById('chat-container');

// Only run the chat logic if we are actually on the chat page (not the login page)
if (chatForm && chatInput && chatContainer) {
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Display user message
        appendMessage('user', query);
        chatInput.value = '';

        // Create a placeholder for the assistant's streaming response
        const assistantBubble = appendMessage('assistant', '');
        
        // Accumulate the full text safely before rendering
        let fullText = ""; 

        try {
            // Fetch with Authorization Header
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('token')}`
                },
                body: JSON.stringify({ query: query })
            });

            // Handle Expired or Invalid Tokens
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/login.html';
                return;
            }

            if (!response.ok) {
                assistantBubble.innerHTML = `<span style="color:red">Error: ${response.statusText}</span>`;
                return;
            }

            // Parse the Server-Sent Events Stream manually
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                
                // SSE chunks are separated by double newlines
                const parts = buffer.split('\n\n');
                buffer = parts.pop(); // Keep the last incomplete chunk in the buffer

                for (const part of parts) {
                    let eventType = 'message';
                    let data = null;

                    // Parse event and data lines
                    const lines = part.split('\n');
                    for (const line of lines) {
                        if (line.startsWith('event: ')) {
                            eventType = line.substring(7).trim();
                        } else if (line.startsWith('data: ')) {
                            const dataString = line.substring(6).trim();
                            // Safely attempt to parse JSON to prevent crashes on fragmented chunks
                            try {
                                data = JSON.parse(dataString);
                            } catch (parseError) {
                                console.warn("Skipping unparseable chunk:", dataString);
                                continue; 
                            }
                        }
                    }

                    // Handle specific events defined in app.py
                    if (eventType === 'meta' && data) {
                        console.log("Routed to:", data.decision.tier);
                    } else if (eventType === 'delta' && data && data.text) {
                        // Append text chunks to the accumulator and re-render
                        fullText += data.text;
                        assistantBubble.innerHTML = escapeHTML(fullText).replace(/\n/g, '<br>');
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

// Helper function to render messages safely
function appendMessage(role, text) {
    if (!chatContainer) return null;
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    msgDiv.innerHTML = text ? escapeHTML(text).replace(/\n/g, '<br>') : '';
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return msgDiv;
}

// Helper function to prevent XSS attacks
function escapeHTML(str) {
    if (!str) return "";
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

// Optional: Attach this to a logout button in your HTML
function logout() {
    localStorage.removeItem('token');
    window.location.href = '/login.html';
}
