import { StyleSheet, Platform } from "react-native";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container:{
        flex: 1,
        backgroundColor: theme.colors.black
    },
    content: {
        width: '100%',
        height: '20%',
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        backgroundColor: theme.colors.black
    },
    cameraButton: {
        height: 70,
        width: 70,
        borderWidth: 5,
        borderColor: theme.colors.white,
        borderRadius: 35,
        backgroundColor: theme.colors.gray,
        justifyContent: 'center',
        alignItems: 'center'
    },
    center: {
        height: 46,
        width: 46,
        backgroundColor: theme.colors.white,
        borderRadius: 23,
    },

    externalCircle: {
        height: 80,
        width: 80,
        borderWidth: 3,
        borderColor: theme.colors.gray,
        borderRadius: 40,
        backgroundColor: 'transparent',
        justifyContent: 'center',
        alignItems: 'center'
    }
})