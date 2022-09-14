import React from "react";
import { Text, View, SafeAreaView, Image } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';

import { PageHeader } from "../../components/PageHeader";
import { Button } from "../../components/Button";
import { Tip } from "../../components/Tip";
import { theme } from "../../global/styles/theme";
import { styles } from "./styles";


export function TipsScreen({navigation}){
    return (
        <View style={styles.container}>
            <PageHeader 
                text={"Get started"}
                onCancelPress={() => navigation.navigate("HomeScreen")}
            />
            <View style={styles.content}>
                <Tip 
                    Icon={() => <MaterialCommunityIcons name="target-variant" size={30} color={theme.colors.primary} />}
                    title={"Make sure skin spot is visible"}
                    text={"Make sure the skin spot isn't covered \nby hair or clothes and you have enough light"}
                />

                <Tip 
                    Icon={() => <MaterialCommunityIcons name="image-filter-center-focus-weak" size={30} color={theme.colors.primary} />}
                    title={"In the middle of the circle"}
                    text={"Keep the skin spot in the center of the white \ncircle"}
                />

                <Tip 
                    Icon={() => <MaterialCommunityIcons name="arrow-top-left-bottom-right" size={30} color={theme.colors.primary}/>}
                    title={"In focus"}
                    text={"Keep the phone 10cm - 20cm from the skin \nspot focus"}
                />
            </View>
            <View style={styles.button}> 
                <Button 
                    text={"Continue"}
                    textColor={theme.colors.white}
                    OnPress={()=>navigation.navigate("CameraScreen")} 
                    extraStyle={{
                        backgroundColor: theme.colors.primary,
                    }}
                />
            </View>
        </View>
    )
}