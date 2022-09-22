import React, { useEffect } from "react";
import { StatusBar, View } from "react-native";
import FlashMessage from "react-native-flash-message";
import { showMessage } from "react-native-flash-message";


import { api } from "./src/services/api";

import Router  from "./src/routes";

export default function App() {

  useEffect(() => {
    async function testingConnection(){
      try{
        await api.get("").then(() => {
          showMessage({ message: "connection established", icon: "success", type: "success"});
          console.log("connection established")})
      }
      catch(err){
        showMessage({ message: "couldn't connect to API", icon: "danger", type: "danger"});
        console.log(err.response)
      }
    }
    testingConnection()
  }, [])

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
