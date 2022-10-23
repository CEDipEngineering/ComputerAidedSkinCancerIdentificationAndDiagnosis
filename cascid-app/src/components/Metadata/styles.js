import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        flex: 1,
        flexDirection: "row",
        backgroundColor: theme.colors.black
    },

    title: {
        color: theme.colors.black

    }
})