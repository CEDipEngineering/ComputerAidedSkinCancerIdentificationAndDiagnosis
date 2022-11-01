import React, { useEffect, useState } from "react"
import {View, SafeAreaView, Image, ActivityIndicator } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';

import { Tip } from "../../components/Tip";
import { PageHeader } from "../../components/PageHeader"
import { Button } from "../../components/Button"
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

import { uploadImage } from "../../services/requests/metadataScreen";
import { hedImage } from "../../services/requests/cameraScreen";

export function PicturePreviewScreen({navigation, route}){

    const {imageData} = route.params
    const [imageHedBase64, setImageHedBase64] = useState(null)
    const [uuid, setUuid] = useState(null)
    const [loading, setLoading] = useState(true)

    async function readImageAsBase64(imagePath) {
        return await FileSystem.readAsStringAsync(
            imagePath, {encoding: FileSystem.EncodingType.Base64})
        }

    useEffect(() => {
        async function getHed(){
            const imageToBase64 = await readImageAsBase64(imageData.uri)
            const uploadImageResponse = await uploadImage(imageToBase64)
            const hed = await hedImage(uploadImageResponse.data.path)
            setImageHedBase64(hed.data.img_base64.split('\'')[1])
            setUuid(uploadImageResponse.data.path)
        }
        console.log("THERE IS SMOETHING")
        getHed()
    },[])

    useEffect(() => {
        if (imageHedBase64){
            setLoading(false)
        }
    },[imageHedBase64])

    return (
        <SafeAreaView style={
            loading ? {justifyContent: "center"} : styles.container}>
            {
                loading ? 
                (
                    <View style={styles.activityIndicator}>
                        <ActivityIndicator size={"large"} color={theme.colors.primary}/>
                    </View>
                )
                :
                (
                <View style={{
                    flex:1}}>
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
                            source={ {uri: 'data:image/jpeg;base64,' + imageHedBase64} }
                        />
                    </View>
                    <View style={styles.button_content}>
                        <Button 
                            text={"continue"}
                            textColor={theme.colors.white}
                            OnPress={()=> 
                                navigation.navigate("MetadataSceen", {
                                    uuid: uuid,
                                    imageHedBase64: imageHedBase64}
                            )}
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
                </View>
                )
            }
            
        </SafeAreaView>
    )
}