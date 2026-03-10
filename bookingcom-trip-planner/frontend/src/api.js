const rawApiBase = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE = rawApiBase
  ? rawApiBase.replace(/\/+$/, '')
  : '/api';

export function apiUrl(path) {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}
