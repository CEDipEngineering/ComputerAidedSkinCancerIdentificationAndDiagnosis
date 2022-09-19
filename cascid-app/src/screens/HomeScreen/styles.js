import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: theme.colors.background,
    },
    content: {
        flexGrow: 1,
        justifyContent: 'flex-end',
        marginHorizontal: metrics.margin,
        marginVertical: iphone ? metrics.margin : 2 * metrics.margin,
    },
    logo: {
        position: "absolute",
        left: metrics.margin,
        height: 200,
        width: 200,
        resizeMode: 'contain'
    },

    textContent: {
        marginBottom: metrics.margin*2
    },

    title: {
        color: theme.colors.primary,
        fontSize: metrics.titleSize,
        marginBottom: metrics.margin/2.5,
        fontWeight: 'bold'
    },
    text: {
        width: '100%',
        textAlign: 'justify',
        color: theme.colors.primary,
        fontSize: metrics.textSize
    },
    buttonContent: {
        width: '100%',
        justifyContent: "space-around",
    },

})