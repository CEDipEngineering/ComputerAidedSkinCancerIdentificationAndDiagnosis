import React from 'react';
import { Text, View, Image, TextInput } from 'react-native';

import { styles } from './styles';

export function ContainerTextInput({
    image,
    title,
    value,
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
                <TextInput
                    style={styles.age}
                    onChangeText={onChange}
                    value={value}
                    placeholder="age"
                    keyboardType="numeric"
                />
            </View>
        </View>
    )
}