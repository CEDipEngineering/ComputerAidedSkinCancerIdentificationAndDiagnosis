import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";

import { HomeScreen } from "../screens/HomeScreen"
import { CameraScreen } from "../screens/CameraScreen";
import { TipsScreen } from "../screens/TipsScreen";
import { PicturePreviewScreen } from "../screens/PicturePreviewScreen";
import { MetadataSceen } from "../screens/MetadataSceen";
import { ResultsScreen } from "../screens/ResultsScreen";

const Stack = createStackNavigator();

export default function Router(){
    return (
        <NavigationContainer>
            <Stack.Navigator
                screenOptions={{
                    headerShown: false
                }}
              >
                <Stack.Screen name="HomeScreen" component={HomeScreen}/>
                <Stack.Screen name="CameraScreen" component={CameraScreen}/>
                <Stack.Screen name="TipsScreen" component={TipsScreen}/>
                <Stack.Screen name="PicturePreviewScreen" component={PicturePreviewScreen}/>
                <Stack.Screen name="MetadataSceen" component={MetadataSceen}/>
                <Stack.Screen name="ResultsScreen" component={ResultsScreen}/>
            </Stack.Navigator>
        </NavigationContainer>
    )
}