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
    previewContainer: {
        flex: 1,
        marginHorizontal: metrics.margin,
        marginVertical: iphone ? metrics.margin : 2 * metrics.margin,
        justifyContent: 'space-between',
        alignItems: 'center'
    },
    photoPreview:{
        width: '100%',
        height: 300,
        borderRadius: metrics.radius
    },
    identified:{
        width: '100%',
        height: 300,
    },
    macros:{
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: "center"
    }
})