import React, { useState } from 'react';
import { Text, View, SafeAreaView, Image } from 'react-native';

import { styles } from './styles';
import { theme } from '../../global/styles/theme';
import { metrics } from '../../global/styles/metrics';

import { Metadata } from '../../components/Metadata';
import logoHorizontal from "../../assets/logo-horizontal.png"
import { Checkbox } from '../../components/CheckBox';

export function MetadataSceen({navigation, route}) {

  //const {imagePreview} = route.imageUri
  console.log("there -> ",route.params)
  const [checked, onChange] = useState(false);

  return (
    <SafeAreaView style={styles.container}>
      <Text>TEST</Text>

      <View style={styles.test}>
        <Image
            style={{
                width: 80,
                height: 80,
                resizeMode: "cover",
                marginTop: metrics.margin,
                borderColor: theme.colors.primary,
                borderWidth: 2
            }}
            source={logoHorizontal}
        />
        <View>
          <Text style={styles.title}>{"test"}</Text>
          <Text style={styles.text}>{"test"}</Text>
        </View>
        <Checkbox 
          checked={checked}
          onChange={onChange}
        />
      </View>
    </SafeAreaView>
  );
}