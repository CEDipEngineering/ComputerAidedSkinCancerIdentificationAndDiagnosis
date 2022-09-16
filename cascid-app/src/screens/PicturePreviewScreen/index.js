import React from "react"
import { Text, View, SafeAreaView, Image } from 'react-native';


import { PageHeader } from "../../components/PageHeader"
import { Button } from "../../components/Button"

export function PicturePreviewScreen({navigation, route}){

    const {imagePreview} = route.params
    return (
        <SafeAreaView>
            <PageHeader 
                text={"Image preview"}
                onCancelPress={() => navigation.goBack()}
            />
            <Image
                style={{
                    width: 100,
                    height: 100
                }}
                source={ {uri: imagePreview.uri} }/>

        </SafeAreaView>
    )
}