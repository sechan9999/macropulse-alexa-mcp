import React, {useState} from 'react';
import {TouchableOpacity, View, Text, StyleSheet} from 'react-native';

interface SignalRowProps {
  ticker: string;
  price: number;
  signal: string; // STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
  score: number;
  hasTVPreferredFocus?: boolean;
}

function colorFor(signal: string): string {
  if (signal.includes('BUY')) {
    return '#34d399';
  }
  if (signal.includes('SELL')) {
    return '#f87171';
  }
  return '#facc15';
}

const SignalRow = ({
  ticker,
  price,
  signal,
  score,
  hasTVPreferredFocus,
}: SignalRowProps) => {
  const [focused, setFocused] = useState(false);
  const color = colorFor(signal);

  return (
    <TouchableOpacity
      hasTVPreferredFocus={hasTVPreferredFocus}
      // react-native-tvos fades a focused TouchableOpacity to activeOpacity (0.2 by
      // default), which made the selected row look disabled; the border marks focus.
      activeOpacity={1}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      style={[styles.row, focused && styles.rowFocused]}>
      <Text style={styles.ticker}>{ticker}</Text>
      <Text style={styles.price}>${price.toFixed(2)}</Text>
      <View style={[styles.badge, {backgroundColor: color}]}>
        <Text style={styles.badgeText}>{signal.replace('_', ' ')}</Text>
      </View>
      <Text style={styles.score}>{score > 0 ? `+${score}` : score}</Text>
    </TouchableOpacity>
  );
};

export default SignalRow;

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a2233',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginBottom: 8,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  rowFocused: {
    borderColor: '#FF9900',
    backgroundColor: '#232f3e',
  },
  ticker: {
    color: '#f8fafc',
    fontSize: 16,
    fontWeight: '700',
    width: 90,
  },
  price: {
    color: '#cbd5e1',
    fontSize: 15,
    width: 90,
  },
  badge: {
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    minWidth: 100,
    alignItems: 'center',
  },
  badgeText: {
    color: '#0b1120',
    fontSize: 12,
    fontWeight: '800',
  },
  score: {
    color: '#94a3b8',
    fontSize: 14,
    marginLeft: 'auto',
  },
});
