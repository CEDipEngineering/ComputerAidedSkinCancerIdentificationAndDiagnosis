import React from "react";
import { Text, View, SafeAreaView, Image } from 'react-native';
import { styles } from "./styles";


export function Tip({Icon, title, text}){
    return (
        <View style={styles.container}>
            <Icon />
            <View style={styles.content}>
                <Text style={styles.title}>{title}</Text>
                <Text style={styles.text}>{text}</Text>
            </View>
        </View>
    )
}