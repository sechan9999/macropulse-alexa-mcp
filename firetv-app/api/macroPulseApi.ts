/**
 * Thin fetch wrapper for rest_server.py (../rest_server.py in the parent repo).
 * Base URL and user_id are configured on the Settings screen and persisted
 * via AsyncStorage — see screens/SettingsScreen.tsx.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL_KEY = 'macropulse.apiBaseUrl';
const USER_ID_KEY = 'macropulse.userId';

// Sensible dev default: 10.0.2.2 is the Android emulator's alias for the
// host machine's localhost. Replace with your rest_server.py's real
// LAN/Cloud Run URL on Settings once you're running on an actual device.
const DEFAULT_API_BASE_URL = 'http://10.0.2.2:8080';

export async function getApiBaseUrl(): Promise<string> {
  const stored = await AsyncStorage.getItem(API_BASE_URL_KEY);
  return stored && stored.trim() ? stored.trim() : DEFAULT_API_BASE_URL;
}

export async function setApiBaseUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(API_BASE_URL_KEY, url.trim());
}

export async function getUserId(): Promise<string> {
  return (await AsyncStorage.getItem(USER_ID_KEY)) ?? '';
}

export async function setUserId(userId: string): Promise<void> {
  await AsyncStorage.setItem(USER_ID_KEY, userId.trim());
}

export interface RegimeResponse {
  as_of: string;
  regime: string;
  regime_score: number;
  sp500: number;
  sp500_mom_pct: number;
  treasury_10y_pct: number;
  yield_curve_slope_pct: number;
  credit_spread_pct: number;
  realized_vol_12m_pct: number;
}

export interface WatchlistSignal {
  ticker: string;
  price: number;
  signal: string;
  score: number;
}

export interface WatchlistSignalsResponse {
  universe: string[];
  signals: WatchlistSignal[];
}

export interface NvdaDangerResponse {
  as_of: string;
  price: number;
  danger_index: number | null;
  danger_label: string;
  rsi: number | null;
  atr_pct: number | null;
  volume_ratio_vs_20d_avg: number | null;
  block_trade_flag: boolean | null;
  vix: number | null;
  peer_returns_pct_over_window: Record<string, number>;
}

async function getJson<T>(path: string): Promise<T> {
  const base = await getApiBaseUrl();
  const res = await fetch(`${base}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export function fetchRegime(): Promise<RegimeResponse> {
  return getJson<RegimeResponse>('/api/regime');
}

export async function fetchWatchlistSignals(
  topN = 5,
): Promise<WatchlistSignalsResponse> {
  const userId = await getUserId();
  const qs = new URLSearchParams({top_n: String(topN)});
  if (userId) {
    qs.set('user_id', userId);
  }
  return getJson<WatchlistSignalsResponse>(
    `/api/watchlist-signals?${qs.toString()}`,
  );
}

export function fetchNvdaDanger(): Promise<NvdaDangerResponse> {
  return getJson<NvdaDangerResponse>('/api/nvda-danger');
}
