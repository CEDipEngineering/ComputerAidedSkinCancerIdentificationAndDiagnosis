import { StatusBar } from 'expo-status-bar';
import { Text, View, SafeAreaView, Image } from 'react-native';

import { styles } from './styles';
import { theme } from '../../global/styles/theme';
import { metrics } from '../../global/styles/metrics';

import { Metadata } from '../../components/Metadata';

export function MetadataSceen({navigation, route}) {

  //const {imagePreview} = route.imageUri
  console.log("there -> ",route.params)

  return (
    <SafeAreaView style={styles.container}>
      <Text>TEST</Text>
      <Metadata />
    </SafeAreaView>
  );
}