import React from 'react';
import { Text, View, Image } from 'react-native';

import { Checkbox } from '../CheckBox';
import { styles } from './styles';

export function ContainerCheckBox({
    image,
    title,
    text,
    checked,
    onChange
}){
    return(
        <View style={styles.container}>
            <Image 
                style={{
                    width: 50,
                    height: 50,
                    resizeMode: "cover"
                }}
                source={image}
            />
            <View style={styles.content}>
                <Text style={styles.title}>{title}</Text>
                <Text style={styles.text}>{text}</Text>
            </View>
            <View style={styles.checkbox}>
                <Checkbox 
                    checked={checked}
                    onChange={onChange}
                />
            </View>
            
        </View>
    )
}