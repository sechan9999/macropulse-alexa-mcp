import React from 'react';
import {NavigationContainer} from '@react-navigation/native';
import LeftHandNav from './navigation/LeftHandNav';

const App = () => {
  return (
    <NavigationContainer>
      <LeftHandNav />
    </NavigationContainer>
  );
};

export default App;
