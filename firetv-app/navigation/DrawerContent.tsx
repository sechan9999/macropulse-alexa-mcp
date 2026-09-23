/**
 * Adapted from AmazonAppDev/hello-world-fire-tv-react-native's
 * navigation/DrawerContent.tsx (MIT-0) — collapse/expand animation and
 * TVEventHandler wiring unchanged; menuItems point at our two screens.
 */
import React, {useState, useRef} from 'react';
import {Animated, StyleSheet} from 'react-native';
import {TVFocusGuideView, useTVEventHandler} from 'react-native';
import DrawerItem from './DrawerItem';

export const COLLAPSED_WIDTH = 60;
const EXPANDED_WIDTH = 170;
const ANIMATION_DURATION = 300;

const menuItems = [
  {name: 'Home', icon: '⚡', screen: 'Home'},
  {name: 'Settings', icon: '⚙️', screen: 'Settings'},
];

interface DrawerContentProps {
  route: string;
}

const DrawerContent = ({route}: DrawerContentProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const widthAnim = useRef(new Animated.Value(COLLAPSED_WIDTH)).current;
  const textOpacityAnim = useRef(new Animated.Value(0)).current;
  const [activeFocus, setActiveFocus] = useState(false);

  const expand = () => {
    setIsExpanded(true);
    setActiveFocus(true);
    Animated.parallel([
      Animated.timing(widthAnim, {
        toValue: EXPANDED_WIDTH,
        duration: ANIMATION_DURATION,
        useNativeDriver: false,
      }),
      Animated.timing(textOpacityAnim, {
        toValue: 1,
        duration: ANIMATION_DURATION,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setActiveFocus(false);
    });
  };

  const collapse = () => {
    Animated.parallel([
      Animated.timing(widthAnim, {
        toValue: COLLAPSED_WIDTH,
        duration: ANIMATION_DURATION,
        useNativeDriver: false,
      }),
      Animated.timing(textOpacityAnim, {
        toValue: 0,
        duration: ANIMATION_DURATION,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setIsExpanded(false);
    });
  };

  useTVEventHandler(({eventType, eventKeyAction}: any) => {
    if (eventKeyAction === 1 && eventType === 'right' && isExpanded) {
      collapse();
    }
  });

  return (
    <Animated.View style={[styles.drawer, {width: widthAnim}]}>
      <TVFocusGuideView trapFocusDown>
        {menuItems.map((item, index) => (
          <DrawerItem
            key={index}
            item={item}
            route={route}
            expand={expand}
            isExpanded={isExpanded}
            textOpacityAnim={textOpacityAnim}
            hasTVPreferredFocus={activeFocus && item.screen === route}
          />
        ))}
      </TVFocusGuideView>
    </Animated.View>
  );
};

export default DrawerContent;

const styles = StyleSheet.create({
  drawer: {
    flex: 1,
    backgroundColor: '#232F3E',
    paddingTop: 50,
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    zIndex: 1,
  },
});
