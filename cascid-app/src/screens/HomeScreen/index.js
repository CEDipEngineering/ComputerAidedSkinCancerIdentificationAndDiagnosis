import { StatusBar } from 'expo-status-bar';
import { Text, View, SafeAreaView, Image } from 'react-native';

import { styles } from './styles';
import { theme } from '../../global/styles/theme';
import { metrics } from '../../global/styles/metrics';

import logoHorizontal from "../../assets/logo-horizontal-bold.png"
import { Button } from '../../components/Button';

export function HomeScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <Image 
        source={logoHorizontal}
        style={styles.logo}
        resizeMode="stretch"
      />

      <View style={styles.content}>
        <View style={styles.textContent}>
          <Text style={styles.title}>Skin cancer diagnosis</Text>
          <Text style={styles.text}>
            We help you by diagnosing possible problems in your skin through artificial intelligence.
          </Text>
        </View>
        <View style={styles.buttonContent}>
          <Button 
            text={"analyze"}
            textColor={theme.colors.white}
            OnPress={()=> console.log("pressed")}
            extraStyle={{
              backgroundColor: theme.colors.primary,
              fontSize: metrics.textSize,
              marginBottom: metrics.margin / 2
          }}

          />
          <Button 
            text={"how does it work?"}
            textColor={theme.colors.white}
            OnPress={()=> console.log("pressed")}
            extraStyle={{
              backgroundColor: theme.colors.gray,
          }}/>
          </View>
      </View>
      
      <StatusBar style="auto" />
    </SafeAreaView>
  );
}