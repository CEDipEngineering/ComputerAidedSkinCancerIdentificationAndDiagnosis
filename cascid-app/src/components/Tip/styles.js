import { StyleSheet } from "react-native";
import { metrics } from "../../global/styles/metrics";
import { theme } from "../../global/styles/theme";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        flexDirection: "row",
        width: "100%",
    },

    content: {
        marginLeft: metrics.margin,
        
    },

    title: {
        fontSize: 20,
        fontWeight: "bold",
    },

    text: {
        textAlign: "justify",
        fontSize: 16,
    },
})