import { StyleSheet, Platform } from "react-native";
import { Dimensions } from 'react-native';

import { theme } from "../../global/styles/theme";

const windowWidth = Dimensions.get('window').width;

export const styles = StyleSheet.create({
    container:{
        flex: 1,
        backgroundColor: theme.colors.black,
        justifyContent: "space-between"
    },

    cameraConfig: {
        width: "100%",
        flex:1, 
        justifyContent: "center",
        alignItems: "center"
    }, 

    menu: {
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

    cameraExternalCircle: {
        height: 200,
        width: 200,
        borderWidth: 3,
        borderColor: theme.colors.gray,
        borderRadius: 100,
        backgroundColor: 'transparent',
        justifyContent: 'center',
        alignItems: 'center',
        opacity: .8
    },

    cameraMiddleCircle:{
        height: 180,
        width: 180,
        borderRadius: 90,
        backgroundColor: theme.colors.white,
        justifyContent: 'center',
        alignItems: 'center',
        opacity: .1
    },

    cameraCentralCircle:{
        position: "absolute",
        height: 10,
        width: 10,
        borderRadius: 5,
        backgroundColor: theme.colors.gray,
    }
})