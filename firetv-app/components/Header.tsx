/**
 * From AmazonAppDev/hello-world-fire-tv-react-native's components/Header.tsx
 * (MIT-0), unchanged.
 */
import React from 'react';
import {TouchableHighlight, Text, StyleSheet} from 'react-native';

interface HeaderProps {
  headerText: string;
}

const Header = ({headerText}: HeaderProps) => {
  return (
    <TouchableHighlight style={styles.headerContainer}>
      <Text style={styles.headerText}>{headerText}</Text>
    </TouchableHighlight>
  );
};

const styles = StyleSheet.create({
  headerContainer: {
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 30,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FF9900',
  },
});

export default Header;
