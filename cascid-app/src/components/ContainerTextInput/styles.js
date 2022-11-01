import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    container: {
        width: "100%",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
    },

    content:{
        height: 100,
        width: "80%",
        justifyContent: "space-around" 
    },

    title:{
        color: theme.colors.black,
        fontSize: 20,
        fontWeight: "bold"
    },

    text:{
        color: theme.colors.black,
        fontSize: 16,
    },

    age:{
        width:"100%",
        borderBottomColor: theme.colors.primary,
        borderBottomWidth: 2
        
    }
})