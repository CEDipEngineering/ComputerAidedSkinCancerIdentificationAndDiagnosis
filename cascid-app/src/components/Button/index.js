
import React from "react";
import { Text, TouchableOpacity, View, ActivityIndicator } from "react-native";
import { metrics } from "../../global/styles/metrics";

import { theme } from "../../global/styles/theme";
import { styles } from './styles'

export function Button({
    text,
    textColor,
    extraStyle,
    OnPress,
    disable,
    loading }) {
    return (
        <TouchableOpacity
            onPress={OnPress}
            style={disable ? [styles.disableButton, extraStyle] : [styles.container, extraStyle]}
            activeOpacity={0.8}
            disabled={disable}
        >
            {
                loading ?
                    <ActivityIndicator size="small" color={theme.colors.white} />
                    :
                    <Text style={{ color: textColor, fontSize: metrics.textSize}}>{text}</Text>
            }
        </TouchableOpacity >
    )
}