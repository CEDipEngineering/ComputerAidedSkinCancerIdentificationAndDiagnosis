import React from "react";
import { createStackNavigator } from '@react-navigation/stack'

import {HomeScreen} from "../screens/HomeScreen"

const { Navigator, Screen } = createStackNavigator()

export function AuthRoutes(){
    return (
        <Navigator
            screenOptions={{
                headerShown: false
            }}
        >
            <Screen 
                name="HomeScreen"
                component={HomeScreen}
            />
        </Navigator>
    )
}