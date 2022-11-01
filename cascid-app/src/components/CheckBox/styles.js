import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

const iphone = Platform.OS === 'ios'

export const styles = StyleSheet.create({
    checkboxBase: {
        width: 30,
        height: 30,
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: metrics.radius,
        borderWidth: 2,
        borderColor: theme.colors.primary,
        backgroundColor: 'transparent',
      },
    
    checkboxChecked: {
        backgroundColor: theme.colors.primary,
    },

    checkboxUnChecked:{
        backgroundColor: theme.colors.white,
    }
})