// Loaded before app.js. For local dev / single-service Render deploys where the
// gateway serves this frontend itself, leave both blank (same-origin, no key
// needed unless you've also set GATEWAY_API_KEY there).
//
// For a split deploy (this file hosted on Vercel, gateway hosted on Render):
window.API_BASE = "";        // e.g. "https://llm-gateway.onrender.com"
window.API_KEY = "";         // must match GATEWAY_API_KEY set on the Render gateway
