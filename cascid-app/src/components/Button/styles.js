import { StyleSheet } from "react-native";
import { metrics } from "../../global/styles/metrics";
import { theme } from "../../global/styles/theme";

export const styles = StyleSheet.create({
    container: {
        width: '100%',
        height: metrics.buttonHeight,
        backgroundColor: theme.colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: metrics.radius,
    },

    disableButton: {
        width: '100%',
        height: metrics.buttonHeight,
        backgroundColor: theme.colors.gray,
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: metrics.radius
    }
})