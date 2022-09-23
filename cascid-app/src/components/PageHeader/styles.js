import { StyleSheet } from "react-native";
import { metrics } from "../../global/styles/metrics";
import { theme } from "../../global/styles/theme";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        marginVertical: iphone ? metrics.margin : 2 * metrics.margin,
        flexDirection: "row",
        height: metrics.buttonHeight,
        alignItems: 'center',
        justifyContent: "space-between",
        borderBottomColor: theme.colors.gray,
        borderBottomWidth: 3,

    },

    title: {
        fontSize: 20,
    },

    border: {
        marginHorizontal: metrics.margin,
        height: 50,
        width: 40,
        backgroundColor: theme.colors.gray
    }
})