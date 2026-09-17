/**
 * Adapted from AmazonAppDev/hello-world-fire-tv-react-native's
 * navigation/DrawerItem.tsx (MIT-0) — the focus/expand/navigate wiring is
 * unchanged; only the icon source (react-native-paper -> emoji, to drop
 * the icon-font dependency) differs.
 */
import React, {useState} from 'react';
import {View, StyleSheet, TouchableOpacity, Animated, Text} from 'react-native';
import {useNavigation, ParamListBase} from '@react-navigation/native';
import {DrawerNavigationProp} from '@react-navigation/drawer';

interface MenuItem {
  name: string;
  icon: string;
  screen: string;
}

interface DrawerItemProps {
  item: MenuItem;
  route: string;
  expand: () => void;
  isExpanded: boolean;
  textOpacityAnim: Animated.Value;
  hasTVPreferredFocus: boolean;
}

const DrawerItem = ({
  item,
  route,
  expand,
  isExpanded,
  textOpacityAnim,
  hasTVPreferredFocus,
}: DrawerItemProps) => {
  const navigation = useNavigation<DrawerNavigationProp<ParamListBase>>();
  const [isFocused, setIsFocused] = useState<boolean>(false);
  const isActive = item.screen === route;

  return (
    <TouchableOpacity
      key={item.name}
      hasTVPreferredFocus={hasTVPreferredFocus}
      onFocus={() => {
        setIsFocused(true);
        if (!isExpanded) {
          expand();
        }
      }}
      onBlur={() => {
        setIsFocused(false);
      }}
      onPress={() => {
        navigation.navigate(item.screen);
      }}>
      <View style={styles.menuItem}>
        <Text style={[styles.icon, isActive && styles.activeIcon]}>
          {item.icon}
        </Text>
        <Animated.Text
          style={[
            styles.menuItemText,
            {opacity: textOpacityAnim},
            isFocused && styles.focusedMenuItem,
          ]}>
          {item.name}
        </Animated.Text>
      </View>
    </TouchableOpacity>
  );
};

export default DrawerItem;

const styles = StyleSheet.create({
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    height: 54,
  },
  icon: {
    fontSize: 22,
    width: 24,
    textAlign: 'center',
    color: 'white',
  },
  activeIcon: {
    color: '#FF9900',
  },
  menuItemText: {
    color: 'white',
    marginLeft: 10,
    paddingBottom: 4,
  },
  focusedMenuItem: {
    borderBottomColor: '#FF9900',
    borderBottomWidth: 2,
  },
});
