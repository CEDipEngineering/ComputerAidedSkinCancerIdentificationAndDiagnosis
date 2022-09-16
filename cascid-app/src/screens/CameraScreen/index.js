import React, { useState, useEffect, useRef } from "react";
import { useIsFocused } from '@react-navigation/native';
import { View, TouchableOpacity, SafeAreaView, Text} from "react-native";
import { Camera } from 'expo-camera';
import { MaterialIcons } from '@expo/vector-icons';
import { Ionicons } from '@expo/vector-icons';

//import { goBack, navigate } from "../../services/navigation";
import { styles } from "./styles";
import { theme } from "../../global/styles/theme";

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
            navigation.navigate("PicturePreviewScreen", {imagePreview: data})
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
            <View style={styles.content}></View>
            {
            isFocused && 
            <Camera
                style={{ 
                    flex: 1,
                    justifyContent: "center",
                    alignItems: "center"
                }}
                type={cameraType} 
                flashMode={handleFlashMode(enableFlash)}
                ref={camRef}>
                    <View style={styles.externalCircle}></View>
                </Camera>
            }
            <View style={styles.content}>
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