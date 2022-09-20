import React from "react"
import {View, SafeAreaView, Image } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import { showMessage } from "react-native-flash-message";

import { Tip } from "../../components/Tip";
import { PageHeader } from "../../components/PageHeader"
import { Button } from "../../components/Button"
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

import { uploadImage, predictImage } from "../../services/requests/picturePreviewScreen";

export function PicturePreviewScreen({navigation, route}){

    const {imagePreview} = route.params

    // functions
    async function readImageAsBase64(imagePath) {
        return await FileSystem.readAsStringAsync(
            imagePath, {encoding: FileSystem.EncodingType.Base64})
    }

    async function sendImage(imagePath){
        try {
            const imageToBase64 = await readImageAsBase64(imagePath)
            const uploadImageResponse = await uploadImage(imageToBase64)
            const predictImageResponse = await predictImage(uploadImageResponse.data.path)
            console.log(predictImageResponse)
            navigation.navigate("ResultsScreen", 
                {imageUri: imagePreview.uri, prediction: predictImageResponse.data})
            
        }
        catch (error){
            console.log(error.response)
            showMessage({ message: "something went wrong", icon: 'danger', type: 'danger'});
        }        
    }

    return (
        <SafeAreaView style={styles.container}>
            <PageHeader 
                text={"Image preview"}
                onCancelPress={() => navigation.navigate("HomeScreen")}
            />
            <View style={styles.content}>
                <Tip 
                    Icon={() => <MaterialCommunityIcons name="eye" size={30} color={theme.colors.primary} />}
                    title={"Check image quality"}
                    text={"Check if you can see the spot and also the image quality"}
                />

                <Image
                    style={{
                        width: 300,
                        height: 300,
                        resizeMode: "cover",
                        marginTop: metrics.margin,
                        borderColor: theme.colors.primary,
                        borderWidth: 2
                    }}
                    source={ {uri: imagePreview.uri} }
                />
            </View>
            <View style={styles.button_content}>
                <Button 
                    text={"analyze"}
                    textColor={theme.colors.white}
                    OnPress={()=> {sendImage(imagePreview.uri)}}
                    extraStyle={{
                    backgroundColor: theme.colors.primary,
                    fontSize: metrics.textSize,
                    marginBottom: metrics.margin / 2
                }}/>
                <Button 
                    text={"take another picture"}
                    textColor={theme.colors.white}
                    OnPress={()=> navigation.goBack()}
                    extraStyle={{
                    backgroundColor: theme.colors.gray,
                }}/>
            </View>
        </SafeAreaView>
    )
}