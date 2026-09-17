import React from 'react';
import {View, Text, StyleSheet} from 'react-native';

interface RegimeCardProps {
  regime: string; // e.g. "Risk-On 🟢", "Neutral 🟡", "Risk-Off 🔴"
  regimeScore: number;
  asOf: string;
}

function colorFor(regime: string): string {
  if (regime.includes('Risk-Off')) {
    return '#f87171';
  }
  if (regime.includes('Risk-On')) {
    return '#34d399';
  }
  return '#facc15';
}

const RegimeCard = ({regime, regimeScore, asOf}: RegimeCardProps) => {
  const color = colorFor(regime);
  return (
    <View style={[styles.card, {borderColor: color}]}>
      <Text style={styles.label}>MACRO REGIME</Text>
      <Text style={[styles.regime, {color}]}>{regime}</Text>
      <Text style={styles.meta}>
        Score {regimeScore.toFixed(2)} · as of {asOf}
      </Text>
    </View>
  );
};

export default RegimeCard;

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#1a2233',
    borderWidth: 2,
    borderRadius: 16,
    padding: 24,
    minWidth: 320,
  },
  label: {
    color: '#94a3b8',
    fontSize: 13,
    fontWeight: '600',
    letterSpacing: 1,
  },
  regime: {
    fontSize: 34,
    fontWeight: '800',
    marginTop: 6,
  },
  meta: {
    color: '#64748b',
    fontSize: 14,
    marginTop: 8,
  },
});
