import React, { useState } from 'react';
import { Text, View, SafeAreaView, Image, ScrollView } from 'react-native';

import { styles } from './styles';
import { theme } from '../../global/styles/theme';
import { showMessage } from "react-native-flash-message";
import * as FileSystem from 'expo-file-system';

import { PageHeader } from '../../components/PageHeader';
import { ContainerCheckBox } from '../../components/ContainerCheckBox';
import { Button } from "../../components/Button"

import { uploadImage, predictImage } from "../../services/requests/picturePreviewScreen";

import CancerHistoryImage from "../../assets/cancer_history.png"
import SkinCancerHistoryImage from "../../assets/skin_cancer_history.png"
import SmokeImage from "../../assets/smoke.png"
import DrinkImage from "../../assets/drink.png"
import PesticideImage from "../../assets/pesticide.png"

export function MetadataSceen({navigation, route}) {

  const [loading, setLoading] = useState(false)
  const [smoke, setSmoke] = useState(false);
  const [drink, setDrink] = useState(false);
  const [pesticide, setPesticide] = useState(false);
  const [cancerHistory, setCancerHistory] = useState(false);
  const [skinCancerHistory, setSkinCancerHistory] = useState(false);

  async function readImageAsBase64(imagePath) {
    return await FileSystem.readAsStringAsync(
        imagePath, {encoding: FileSystem.EncodingType.Base64})
  }

  async function sendImage(imagePath){
    try {
        setLoading(true)
        const imageToBase64 = await readImageAsBase64(imagePath)
        const uploadImageResponse = await uploadImage(imageToBase64)
        const predictImageResponse = await predictImage(uploadImageResponse.data.path)
        navigation.navigate("ResultsScreen", 
            {   
                imageUri: imagePath, 
                prediction: predictImageResponse.data.report
            })
        setLoading(false)
        }
    catch (error){
      setLoading(false)
      console.log(error.message)
      showMessage({ message: "something went wrong", icon: 'danger', type: 'danger'});
    }        
  }

  return (
    <SafeAreaView style={styles.container}>
      <PageHeader 
          text={"Characteristics"}
          onCancelPress={() => navigation.goBack()}
      />
      <ScrollView>
      <ContainerCheckBox 
        image={SmokeImage}
        title={"smoke"}
        text={"lorem ipsum lorem ipsum lorem ipsum lorem ipsum lorem ipsum"}
        checked={smoke}
        onChange={setSmoke}
      />

      <ContainerCheckBox 
        image={DrinkImage}
        title={"drink"}
        text={"lorem ipsum lorem ipsum lorem ipsum lorem ipsum lorem ipsum"}
        checked={drink}
        onChange={setDrink}
      />

      <ContainerCheckBox 
        image={PesticideImage}
        title={"pesticide"}
        text={"lorem ipsum lorem ipsum lorem ipsum lorem ipsum lorem ipsum"}
        checked={pesticide}
        onChange={setPesticide}
      />

      <ContainerCheckBox 
        image={CancerHistoryImage}
        title={"cancer history"}
        text={"lorem ipsum lorem ipsum lorem ipsum lorem ipsum lorem ipsum"}
        checked={cancerHistory}
        onChange={setCancerHistory}
      />

      <ContainerCheckBox 
        image={SkinCancerHistoryImage}
        title={"skin cancer history"}
        text={"lorem ipsum lorem ipsum lorem ipsum lorem ipsum lorem ipsum"}
        checked={skinCancerHistory}
        onChange={setSkinCancerHistory}
      />
      </ScrollView>
      <Button 
          text={"continue"}
          textColor={theme.colors.white}
          OnPress={()=> sendImage(route.params.imageUri)}
          loading={loading}
          extraStyle={{
            backgroundColor: theme.colors.primary,
      }}/>
    </SafeAreaView>
  );
}