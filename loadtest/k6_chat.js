// k6 load test for POST /api/chat (non-streaming path - easiest to assert on).
// Ramps per the plan's "10 -> 100 -> 1,000 concurrent users" (scaled down to what
// a free Groq tier and a laptop can survive; raise MAX_VUS if you have headroom).
//
// Install k6:  sudo apt install k6   (or: https://k6.io/docs/get-started/installation)
// Run:         k6 run loadtest/k6_chat.js
// Against a deployed gateway:
//              BASE_URL=https://your-gateway.onrender.com API_KEY=your-key k6 run loadtest/k6_chat.js
//
// Each VU registers + logs in its own account on first use (cached per VU for
// the rest of the run) rather than sharing one login - now required since
// /api/chat is authenticated, and avoids one account's "free" policy quota
// (50 queries/month) being exhausted and downgrading every request mid-test.

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const API_KEY = __ENV.API_KEY || "";
const MAX_VUS = Number(__ENV.MAX_VUS || 100);
const PASSWORD = "LoadTest123!";

const routerLatency = new Trend("router_latency_ms");
const genLatency = new Trend("generation_latency_ms");
const highTierShare = new Rate("routed_to_high_tier");
const authFailRate = new Rate("auth_failed");

const PROMPTS = [
  "hi",
  "What is the capital of Japan?",
  "Summarize the benefits of exercise in two sentences.",
  "Write a Python function to reverse a linked list and explain the time complexity.",
  "Prove that there are infinitely many prime numbers.",
];

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: Math.min(10, MAX_VUS) },
        { duration: "1m", target: Math.min(10, MAX_VUS) },
        { duration: "30s", target: Math.min(100, MAX_VUS) },
        { duration: "1m", target: Math.min(100, MAX_VUS) },
        { duration: "30s", target: MAX_VUS },
        { duration: "1m", target: MAX_VUS },
        { duration: "30s", target: 0 },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.05"],       // <5% errors
    http_req_duration: ["p(95)<8000"],    // p95 under 8s (generation-bound, not router-bound)
  },
};

// Per-VU token cache. k6 reuses the same VU (and its JS heap) across iterations
// within a scenario, so this persists for the life of that VU without needing
// a shared-state workaround.
const tokenByVU = {};

function getToken() {
  const vu = String(__VU);
  if (tokenByVU[vu]) return tokenByVU[vu];

  const email = `loadtest-vu${vu}-${Date.now()}@test.local`;
  http.post(`${BASE_URL}/api/auth/register`, JSON.stringify({ email, password: PASSWORD }),
    { headers: { "Content-Type": "application/json" } });
  const loginRes = http.post(`${BASE_URL}/api/auth/login`, JSON.stringify({ email, password: PASSWORD }),
    { headers: { "Content-Type": "application/json" } });

  let token = "";
  try {
    token = JSON.parse(loginRes.body).access_token || "";
  } catch {
    /* leave token empty; authFailRate below will flag it */
  }
  authFailRate.add(!token);
  tokenByVU[vu] = token;
  return token;
}

export default function () {
  const query = PROMPTS[Math.floor(Math.random() * PROMPTS.length)];
  const token = getToken();
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (API_KEY) headers["X-API-Key"] = API_KEY;

  const res = http.post(
    `${BASE_URL}/api/chat`,
    JSON.stringify({ query, use_cache: false }),
    { headers, timeout: "30s" }
  );

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has an answer": (r) => {
      try {
        return !!JSON.parse(r.body).answer;
      } catch {
        return false;
      }
    },
  });

  if (res.status === 200) {
    try {
      const body = JSON.parse(res.body);
      routerLatency.add(body.router_latency_ms);
      genLatency.add(body.generation_latency_ms);
      highTierShare.add(body.decision.tier === "high");
    } catch {
      /* ignore parse errors in metrics-only path */
    }
  }

  sleep(1);
}
