import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";

import { HomeScreen } from "../screens/HomeScreen"
import { CameraScreen } from "../screens/CameraScreen";

const Stack = createStackNavigator();

export default function Router(){
    return (
        <NavigationContainer>
            <Stack.Navigator
                screenOptions={{
                    headerShown: false
                }}
              >
                <Stack.Screen 
                    name="HomeScreen"
                    component={HomeScreen}
                    opti
                    />
                <Stack.Screen name="CameraScreen" component={CameraScreen}/>
            </Stack.Navigator>
        </NavigationContainer>
    )
}