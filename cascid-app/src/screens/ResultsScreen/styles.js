import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";
import { Dimensions } from 'react-native';

const windowHeight = Dimensions.get('window').height;

export const styles = StyleSheet.create({
    container: {
        flex:1,
        padding: metrics.margin,
        backgroundColor: theme.colors.background
    },
    content: {
        alignItems: "center",
    },

    tips: {
        width: "100%",
        justifyContent: "space-between",
        height: 120,
        marginBottom: metrics.margin
    },

    button_content: {
        width: "100%",
        position: "absolute",
        left: metrics.margin,
        bottom: metrics.margin
    }

})