import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        flex: 1,
        padding: metrics.margin,
        backgroundColor: theme.colors.white,
    },

    scroll_view:{
        width: "100%"
    },
})