import React from "react"
import {View, SafeAreaView, Image } from 'react-native';
import { MaterialCommunityIcons, FontAwesome5, Ionicons } from '@expo/vector-icons';


import { Tip } from "../../components/Tip";
import { PageHeader } from "../../components/PageHeader"
import { Button } from "../../components/Button"
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

export function ResultsScreen({navigation, route}){

    return (
        <SafeAreaView style={styles.container}>
            <PageHeader 
                text={"Diagnostic"}
                onCancelPress={() => navigation.navigate("HomeScreen")}
            />
            <View style={styles.content}>
                <Image
                    style={{
                        width: 300,
                        height: 300,
                        resizeMode: "cover",
                        marginVertical: metrics.margin,
                        borderColor: theme.colors.primary,
                        borderWidth: 2
                    }}
                    source={ {uri: route.params.imageUri} }
                />

                <View style={styles.tips}>
                    <Tip 
                        Icon={() => <Ionicons name="newspaper-outline" size={30} color={theme.colors.primary} />}
                        title={"Diagnostic"}
                        text={JSON.stringify(route.params.prediction)}
                    />
                    <Tip 
                        Icon={() => <MaterialCommunityIcons name="chart-bell-curve-cumulative" size={30} color={theme.colors.primary} />}
                        title={"Conffidence"}
                        text={"82%"}
                    />
                </View>
            </View>
            <View style={styles.button_content}>
                <Button 
                    text={"done"}
                    textColor={theme.colors.white}
                    OnPress={()=> navigation.navigate("HomeScreen")}
                    extraStyle={{
                    backgroundColor: theme.colors.primary,
                }}/>
            </View>
        </SafeAreaView>
    )
}