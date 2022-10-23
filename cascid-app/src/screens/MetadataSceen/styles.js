import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        marginTop: 300,
        flex: 1,
        backgroundColor: theme.colors.background,
    },

    test: {
        width: "100%",
        flexDirection: "row",
        backgroundColor: theme.colors.gray
    },

    title: {
        color: theme.colors.black

    }
})