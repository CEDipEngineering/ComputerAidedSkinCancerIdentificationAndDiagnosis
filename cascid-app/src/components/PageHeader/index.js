import React from "react";

import { SafeAreaView, Text, View } from 'react-native';
import { Feather } from '@expo/vector-icons';

import { theme } from "../../global/styles/theme";
import { styles } from "./styles";

export function PageHeader({text, onCancelPress, color=theme.colors.black}){
    return (
        <SafeAreaView style={styles.container}>
            <Text style={[styles.title, {color: color}]}>{text}</Text>
              <Feather
                name="x-circle"
                size={35}
                color={color}
                onPress={() => onCancelPress()}
            />
            
        </SafeAreaView>
    )
}