import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';

import { styles } from './styles';

export function Checkbox({
  checked,
  onChange
}) {

  function onCheckmarkPress() {
    onChange(!checked);
  }

  const iconProps = checked ? styles.checkboxChecked : styles.checkboxChecked;

  return (
    <Pressable
      style={[
        styles.checkboxBase,
        checked
          ? styles.checkboxChecked
          : styles.checkboxUnChecked,
      ]}
      onPress={onCheckmarkPress}>
      {checked && (
        <Ionicons
          name="checkmark"
          size={24}
          color="white"
          {...iconProps}
        />
      )}
    </Pressable>
  );
}