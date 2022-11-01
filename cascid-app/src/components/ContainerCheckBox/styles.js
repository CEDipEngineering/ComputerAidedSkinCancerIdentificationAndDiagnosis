import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        width: "100%",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        padding: metrics.padding,
    },

    content:{
        height: 100,
        width: "60%",
        alignItems: "flex-start"
    },

    title:{
        color: theme.colors.black,
        fontSize: 20,
        fontWeight: "bold"
    },

    text:{
        color: theme.colors.black,
        fontSize: 16,
    },

    checkbox:{
        marginLeft: metrics.margin
    }
})