import React, {useCallback, useEffect, useState} from 'react';
import {
  SafeAreaView,
  ScrollView,
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import {Header, RegimeCard, StatTile, SignalRow} from '../components';
import {
  fetchRegime,
  fetchWatchlistSignals,
  fetchNvdaDanger,
  RegimeResponse,
  WatchlistSignal,
  NvdaDangerResponse,
} from '../api/macroPulseApi';

const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes — glanceable, not real-time

const HomeScreen = () => {
  const [regime, setRegime] = useState<RegimeResponse | null>(null);
  const [signals, setSignals] = useState<WatchlistSignal[]>([]);
  const [nvda, setNvda] = useState<NvdaDangerResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAll = useCallback(async () => {
    try {
      const [regimeRes, signalsRes, nvdaRes] = await Promise.all([
        fetchRegime(),
        fetchWatchlistSignals(5),
        fetchNvdaDanger(),
      ]);
      setRegime(regimeRes);
      setSignals(signalsRes.signals);
      setNvda(nvdaRes);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
    const id = setInterval(loadAll, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [loadAll]);

  return (
    <SafeAreaView style={styles.container}>
      <Header headerText="Macro Pulse" />

      {loading && !regime ? (
        <ActivityIndicator size="large" color="#FF9900" style={styles.loader} />
      ) : (
        <ScrollView contentContainerStyle={styles.content}>
          {error ? (
            <Text style={styles.error}>
              Couldn't reach the Macro Pulse API ({error}). Check the URL on the
              Settings screen.
            </Text>
          ) : null}

          {regime ? (
            <View style={styles.topRow}>
              <RegimeCard
                regime={regime.regime}
                regimeScore={regime.regime_score}
                asOf={regime.as_of}
              />
              <View style={styles.statGrid}>
                <StatTile
                  label="S&P 500"
                  value={regime.sp500.toLocaleString()}
                  delta={`${regime.sp500_mom_pct >= 0 ? '+' : ''}${regime.sp500_mom_pct.toFixed(2)}% MoM`}
                  deltaPositive={regime.sp500_mom_pct >= 0}
                />
                <StatTile
                  label="10Y Yield"
                  value={`${regime.treasury_10y_pct.toFixed(2)}%`}
                />
                <StatTile
                  label="Yield Curve"
                  value={`${regime.yield_curve_slope_pct.toFixed(2)}%`}
                />
                <StatTile
                  label="12M Realized Vol"
                  value={`${regime.realized_vol_12m_pct.toFixed(1)}%`}
                />
              </View>
            </View>
          ) : null}

          {nvda ? (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>NVDA Danger Zone</Text>
              <StatTile
                label={nvda.danger_label}
                value={`$${nvda.price.toFixed(2)}`}
                delta={`danger index ${nvda.danger_index?.toFixed(2) ?? '—'}`}
              />
            </View>
          ) : null}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Watchlist Signals</Text>
            {signals.length === 0 ? (
              <Text style={styles.empty}>
                No signals yet — set a user_id with a saved watchlist on
                Settings, or wait for the default watchlist to load.
              </Text>
            ) : (
              signals.map((s, i) => (
                <SignalRow
                  key={s.ticker}
                  ticker={s.ticker}
                  price={s.price}
                  signal={s.signal}
                  score={s.score}
                  hasTVPreferredFocus={i === 0}
                />
              ))
            )}
          </View>
        </ScrollView>
      )}
    </SafeAreaView>
  );
};

export default HomeScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#12181F',
  },
  loader: {
    flex: 1,
  },
  content: {
    padding: 32,
    paddingBottom: 64,
  },
  topRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 32,
  },
  statGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    flex: 1,
  },
  section: {
    marginBottom: 32,
  },
  sectionTitle: {
    color: '#f8fafc',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 12,
  },
  error: {
    color: '#f87171',
    marginBottom: 16,
  },
  empty: {
    color: '#64748b',
  },
});
