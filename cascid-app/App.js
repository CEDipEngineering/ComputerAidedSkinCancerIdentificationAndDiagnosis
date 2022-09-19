import React from "react";
import { StatusBar, View } from "react-native";
import FlashMessage from "react-native-flash-message";

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
      <FlashMessage position="top" style={{ marginTop: 40 }} />
    </View>
  );
}
