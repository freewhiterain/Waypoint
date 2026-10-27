import type { Conversation, HistoryMessage, StreamEvent, User } from "./types";

const TOKEN_KEY = "waypoint.token";
const USER_KEY = "waypoint.user";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export function loadSession(): { token: string; user: User | null } {
  try {
    const token = localStorage.getItem(TOKEN_KEY) || "";
    const raw = localStorage.getItem(USER_KEY);
    return { token, user: raw ? (JSON.parse(raw) as User) : null };
  } catch {
    return { token: "", user: null };
  }
}

export function saveSession(token: string, user: User | null) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  } catch {
    /* 隐私模式下 localStorage 可能不可用，会话只在内存里保持 */
  }
}

let authToken = loadSession().token;
let onUnauthorized: () => void = () => {};

export function setAuthToken(token: string) {
  authToken = token;
}

export function setUnauthorizedHandler(handler: () => void) {
  onUnauthorized = handler;
}

function headers(json = true): HeadersInit {
  const result: Record<string, string> = {};
  if (json) result["Content-Type"] = "application/json";
  if (authToken) result.Authorization = `Bearer ${authToken}`;
  return result;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {
    /* 非 JSON 错误体 */
  }
  return `请求失败（${response.status}）`;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: headers(body !== undefined),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 401) onUnauthorized();
  if (!response.ok) throw new ApiError(await errorMessage(response), response.status);
  return (await response.json()) as T;
}

interface TokenResponse {
  access_token: string;
  user: User;
}

export const api = {
  login: (username: string, password: string) =>
    request<TokenResponse>("POST", "/api/v1/users/login", { username, password }),
  register: (username: string, email: string, password: string) =>
    request<TokenResponse>("POST", "/api/v1/users/register", { username, email, password }),
  listConversations: () => request<Conversation[]>("GET", "/api/v1/conversations"),
  createConversation: (title = "新的旅程") =>
    request<Conversation & { initial_message: HistoryMessage }>("POST", "/api/v1/conversations", { title }),
  renameConversation: (id: string, title: string) =>
    request<Conversation>("PATCH", `/api/v1/conversations/${id}`, { title }),
  deleteConversation: (id: string) => request<unknown>("DELETE", `/api/v1/conversations/${id}`),
  history: (id: string) =>
    request<{ conversation: Conversation; messages: HistoryMessage[] }>("GET", `/api/v1/chat/history/${id}`),
};

/**
 * 读取 POST 返回的 SSE 流。帧可能被网络切成任意片段，按空行分帧后再解析。
 */
export async function streamPost(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify(body),
    signal,
  });
  if (response.status === 401) onUnauthorized();
  if (!response.ok || !response.body) throw new ApiError(await errorMessage(response), response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) {
        try {
          onEvent(JSON.parse(data) as StreamEvent);
        } catch {
          /* 忽略无法解析的帧 */
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
