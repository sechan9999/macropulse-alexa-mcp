import React, {useEffect, useState} from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
} from 'react-native';
import {Header} from '../components';
import {
  getApiBaseUrl,
  setApiBaseUrl,
  getUserId,
  setUserId,
} from '../api/macroPulseApi';

const SettingsScreen = () => {
  const [apiBaseUrl, setApiBaseUrlInput] = useState('');
  const [userId, setUserIdInput] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    (async () => {
      setApiBaseUrlInput(await getApiBaseUrl());
      setUserIdInput(await getUserId());
    })();
  }, []);

  const onSave = async () => {
    await setApiBaseUrl(apiBaseUrl);
    await setUserId(userId);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <SafeAreaView style={styles.container}>
      <Header headerText="Settings" />
      <View style={styles.content}>
        <Text style={styles.label}>Macro Pulse API URL</Text>
        <Text style={styles.hint}>
          rest_server.py's base URL. Use 10.0.2.2 for the host machine from an
          Android emulator, or your Cloud Run URL for the deployed API.
        </Text>
        <TextInput
          style={styles.input}
          value={apiBaseUrl}
          onChangeText={setApiBaseUrlInput}
          placeholder="http://10.0.2.2:8080"
          placeholderTextColor="#64748b"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text style={[styles.label, styles.labelSpaced]}>
          User ID (optional)
        </Text>
        <Text style={styles.hint}>
          If set, the Watchlist Signals list uses this user's saved Firestore
          watchlist instead of the shared default universe.
        </Text>
        <TextInput
          style={styles.input}
          value={userId}
          onChangeText={setUserIdInput}
          placeholder="e.g. alexa-user-123"
          placeholderTextColor="#64748b"
          autoCapitalize="none"
          autoCorrect={false}
        />

        <TouchableOpacity
          style={styles.saveButton}
          onPress={onSave}
          hasTVPreferredFocus>
          <Text style={styles.saveButtonText}>
            {saved ? 'Saved ✓' : 'Save'}
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

export default SettingsScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#12181F',
  },
  content: {
    padding: 32,
    maxWidth: 560,
  },
  label: {
    color: '#f8fafc',
    fontSize: 16,
    fontWeight: '700',
  },
  labelSpaced: {
    marginTop: 24,
  },
  hint: {
    color: '#64748b',
    fontSize: 13,
    marginTop: 4,
    marginBottom: 10,
  },
  input: {
    backgroundColor: '#1a2233',
    borderRadius: 8,
    padding: 12,
    color: '#f8fafc',
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#334155',
  },
  saveButton: {
    marginTop: 32,
    backgroundColor: '#FF9900',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#0b1120',
    fontSize: 16,
    fontWeight: '800',
  },
});
