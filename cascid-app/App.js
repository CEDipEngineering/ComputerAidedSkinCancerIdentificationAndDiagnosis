import React from "react";
import { StatusBar, View } from "react-native";

import Router  from "./src/routes";

export default function App() {
  return (
    <View style={{flex: 1}}>
      <StatusBar
        barStyle="light-content"
        backgroundColor="transparent"
        translucent
      >
      </StatusBar>
      <Router/>
    </View>
  );
}
