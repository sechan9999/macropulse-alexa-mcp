/**
 * Adapted from AmazonAppDev/hello-world-fire-tv-react-native's
 * navigation/LeftHandNav.tsx (MIT-0) — same permanent-drawer setup,
 * pointed at our two screens instead of Home/Movies/TVShows/Settings.
 */
import React from 'react';
import {
  createDrawerNavigator,
  DrawerContentComponentProps,
} from '@react-navigation/drawer';
import {HomeScreen, SettingsScreen} from '../screens';
import DrawerContent from './DrawerContent';

const Drawer = createDrawerNavigator();

const LeftHandNav = () => {
  return (
    <Drawer.Navigator
      drawerContent={(props: DrawerContentComponentProps) => {
        const {state} = props;
        const currentRoute = props.state.routeNames[state.index];
        return <DrawerContent route={currentRoute} />;
      }}
      screenOptions={{
        drawerType: 'permanent',
        drawerStyle: {
          width: 'auto',
        },
        headerShown: false,
      }}>
      <Drawer.Screen name="Home" component={HomeScreen} />
      <Drawer.Screen name="Settings" component={SettingsScreen} />
    </Drawer.Navigator>
  );
};

export default LeftHandNav;
