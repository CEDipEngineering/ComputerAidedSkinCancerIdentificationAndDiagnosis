import React, { useState, useEffect, useRef } from "react";
import { useIsFocused } from '@react-navigation/native';
import { View, TouchableOpacity, SafeAreaView, Text} from "react-native";
import { Camera } from 'expo-camera';
import { MaterialIcons } from '@expo/vector-icons';
import { Ionicons } from '@expo/vector-icons';


import { PageHeader } from "../../components/PageHeader"
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";
import { metrics } from "../../global/styles/metrics";

export function CameraScreen({navigation}) {
    const camRef = useRef(null)
    const [hasPermission, setHasPermission] = useState('pending');
    const [cameraType, setCameraType] = useState(Camera.Constants.Type.back);
    const [enableFlash, setEnableFlash] = useState(false)

    const isFocused = useIsFocused() //enables camera on the right moment

    //functions
    function handleFlashMode(flash) {
        if (flash) return 'on'
        return 'off'
    }

    async function takePicture(camRef) {
        if (camRef) {
            const data = await camRef.current.takePictureAsync()
            navigation.navigate("PicturePreviewScreen", {imageData: data})
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