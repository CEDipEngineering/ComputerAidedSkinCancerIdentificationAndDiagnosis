import React, { useState } from 'react';
import { View, SafeAreaView, ScrollView } from 'react-native';

import { styles } from './styles';
import { theme } from '../../global/styles/theme';
import { showMessage } from "react-native-flash-message";
import * as FileSystem from 'expo-file-system';

import { PageHeader } from '../../components/PageHeader';
import { ContainerCheckBox } from '../../components/ContainerCheckBox';
import { ContainerTextInput } from '../../components/ContainerTextInput';
import { Button } from "../../components/Button"

import { uploadImage, predictImage } from "../../services/requests/metadataScreen";

import CancerHistoryImage from "../../assets/cancer_history.png"
import SkinCancerHistoryImage from "../../assets/skin_cancer_history.png"
import SmokeImage from "../../assets/smoke.png"
import DrinkImage from "../../assets/drink.png"
import PesticideImage from "../../assets/pesticide.png"
import AgeImage from "../../assets/age.png"
import { metrics } from '../../global/styles/metrics';

export function MetadataSceen({navigation, route}) {

  const {uuid, imageHedBase64} = route.params

  const [loading, setLoading] = useState(false)
  const [smoke, setSmoke] = useState(false);
  const [drink, setDrink] = useState(false);
  const [pesticide, setPesticide] = useState(false);
  const [cancerHistory, setCancerHistory] = useState(false);
  const [skinCancerHistory, setSkinCancerHistory] = useState(false);
  const [age, setAge] = useState(null);

  async function sendImage(uuid){
    try {
        setLoading(true)
        const metadata = {
          "smoke": smoke,
          "drink": drink,
          "pesticide": pesticide,
          "cancer_history": cancerHistory,
          "skin_cancer_history": skinCancerHistory,
          "age": age
        }
        const predictImageResponse = await predictImage(uuid, metadata)
        
        navigation.navigate("ResultsScreen", 
            {   
                imageUri: imageHedBase64, 
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
      <ScrollView contentContainerStyle={styles.scroll_view}>
        <ContainerTextInput
          image={AgeImage}
          title={"age"}
          value={age}
          onChange={setAge}
        />

        <ContainerCheckBox 
          image={SmokeImage}
          title={"smoke"}
          text={"Do you have a history of smoking?"}
          checked={smoke}
          onChange={setSmoke}
        />

        <ContainerCheckBox 
          image={DrinkImage}
          title={"drink"}
          text={"Do you have a history of drinking?"}
          checked={drink}
          onChange={setDrink}
        />

        <ContainerCheckBox 
          image={PesticideImage}
          title={"pesticide"}
          text={"Do you have a history of exposure to pesticides?"}
          checked={pesticide}
          onChange={setPesticide}
        />

        <ContainerCheckBox 
          image={CancerHistoryImage}
          title={"cancer history"}
          text={"Have you ever had any form of cancer?"}
          checked={cancerHistory}
          onChange={setCancerHistory}
        />

        <ContainerCheckBox 
          image={SkinCancerHistoryImage}
          title={"skin cancer history"}
          text={"Have you ever had skin cancer?"}
          checked={skinCancerHistory}
          onChange={setSkinCancerHistory}
        />
        </ScrollView>
        <Button 
            text={"continue"}
            textColor={theme.colors.white}
            OnPress={()=> sendImage(uuid)}
            loading={loading}
            disable={age <= 0}
            extraStyle={{
              marginVertical: metrics.margin
        }}/>
    </SafeAreaView>
  );
}