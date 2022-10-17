import React from "react";
import { Text, View, Image } from "react-native";

import { styles } from "./styles";
import { metrics } from "../../global/styles/metrics";
import { theme } from "../../global/styles/theme";

export function Metadata({title, text}) {
    <View style={styles.container}> 
        <Image
          style={{
              width: 80,
              height: 800,
              resizeMode: "cover",
              marginTop: metrics.margin,
              borderColor: theme.colors.primary,
              borderWidth: 2
          }}
          source={"test"}
      />
      <View>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.text}>{text}</Text>
      </View>
      </View>
    }