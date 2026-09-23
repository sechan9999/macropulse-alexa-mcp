import React from 'react';
import {View, Text, StyleSheet} from 'react-native';

interface StatTileProps {
  label: string;
  value: string;
  delta?: string;
  deltaPositive?: boolean;
  deltaColor?: string; // overrides the green/red deltaPositive colouring
}

const StatTile = ({label, value, delta, deltaPositive, deltaColor}: StatTileProps) => {
  return (
    <View style={styles.tile}>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.value}>{value}</Text>
      {delta ? (
        <Text
          style={[
            styles.delta,
            {color: deltaColor ?? (deltaPositive ? '#34d399' : '#f87171')},
          ]}>
          {delta}
        </Text>
      ) : null}
    </View>
  );
};

export default StatTile;

const styles = StyleSheet.create({
  tile: {
    backgroundColor: '#1a2233',
    borderRadius: 12,
    padding: 16,
    minWidth: 150,
  },
  label: {
    color: '#94a3b8',
    fontSize: 12,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  value: {
    color: '#f8fafc',
    fontSize: 24,
    fontWeight: '700',
    marginTop: 4,
  },
  delta: {
    fontSize: 13,
    marginTop: 4,
  },
});
