import { StyleSheet } from "react-native";

import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";
import { Dimensions } from 'react-native';

const windowHeight = Dimensions.get('window').height;


export const styles = StyleSheet.create({
    container: {
        flex:1,
        padding: metrics.margin,
        backgroundColor: theme.colors.background
    },
    content: {
      height: 250,
      justifyContent: "space-between",

    },
    button:{
        width: "100%",
        position: "absolute",
        bottom: metrics.margin,
        left: metrics.margin
    }

})