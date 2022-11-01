import React, { useState, useEffect, useRef } from "react";
import { useIsFocused } from '@react-navigation/native';
import { View, TouchableOpacity, SafeAreaView, Text} from "react-native";
import { Camera } from 'expo-camera';
import { MaterialIcons } from '@expo/vector-icons';
import { Ionicons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';

import { PageHeader } from "../../components/PageHeader"
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

import { uploadImage } from "../../services/requests/metadataScreen";
import { hedImage } from "../../services/requests/cameraScreen";

export function CameraScreen({navigation}) {
    const camRef = useRef(null)
    const [hasPermission, setHasPermission] = useState('pending');
    const [cameraType, setCameraType] = useState(Camera.Constants.Type.back);
    const [enableFlash, setEnableFlash] = useState(false)

    const isFocused = useIsFocused() //enables camera on the right moment

    async function readImageAsBase64(imagePath) {
    return await FileSystem.readAsStringAsync(
        imagePath, {encoding: FileSystem.EncodingType.Base64})
    }

    //functions
    function handleFlashMode(flash) {
        if (flash) return 'on'
        return 'off'
    }

    async function takePicture(camRef) {
        if (camRef) {
            const data = await camRef.current.takePictureAsync()
            const imageToBase64 = await readImageAsBase64(data.uri)
            const uploadImageResponse = await uploadImage(imageToBase64)
            const hed = hedImage(uploadImageResponse.data.path)
            console.log(hed)
            //navigation.navigate("PicturePreviewScreen", {imagePreview: data})
        }
    }

    //effects
    useEffect(() => {
        (
            async () => {
                const { status } = await Camera.requestCameraPermissionsAsync();
                setHasPermission(status);
            }
        )();
    }, []);

    useEffect(() => {
        if (hasPermission === 'denied') {
            navigation.navigate('HomeScreen')
        }
    }, [hasPermission]);

    return (
        <SafeAreaView style={styles.container}>
            <View style={{marginHorizontal: metrics.margin}}>
                <PageHeader 
                   text={"Take picture"}
                    onCancelPress={() => navigation.navigate("HomeScreen")}
                    color={theme.colors.white}
                />
            </View>
            {
            isFocused && 
            <Camera
                style={styles.cameraConfig}
                type={cameraType} 
                flashMode={handleFlashMode(enableFlash)}
                ref={camRef}>
                    <View style={styles.cameraExternalCircle}>
                        <View style={styles.cameraMiddleCircle}></View>
                        <View style={styles.cameraCentralCircle}></View>
                    </View>
                </Camera>
            }
            <View style={styles.menu}>
                <MaterialIcons
                    name='flip-camera-android'
                    size={24}
                    color={theme.colors.white}
                    onPress={() => {
                        setCameraType(
                            cameraType === Camera.Constants.Type.back
                                ? Camera.Constants.Type.front
                                : Camera.Constants.Type.back
                        );
                    }}
                />
                <TouchableOpacity
                    style={styles.cameraButton}
                    onPress={async () => takePicture(camRef)}
                >
                    <View style={styles.center} />
                </TouchableOpacity>

                <Ionicons
                    name={enableFlash ? 'flash' : 'flash-off'}
                    size={24} color={theme.colors.white}
                    onPress={() => setEnableFlash(!enableFlash)}
                />
            </View>
        </SafeAreaView>
    )
}