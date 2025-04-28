import { initializeApp } from 'firebase/app';
import { getFirestore } from '@firebase/firestore';
import { getStorage } from 'firebase/storage';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDab-7PWXBoeVljVcuQcP-Y4wu-4v6Vh64",
  authDomain: "dataviz-6f629.firebaseapp.com",
  projectId: "dataviz-6f629",
  storageBucket: "dataviz-6f629.firebasestorage.app",
  messagingSenderId: "722499433834",
  appId: "1:722499433834:web:67aa5a61ff333a7b05e113",
  measurementId: "G-8DEL8SW8N6"
};

// Initialize Firebase with error handling
let app;
let db;
let storage;

try {
  app = initializeApp(firebaseConfig);
  db = getFirestore(app);
  storage = getStorage(app);
  console.log('Firebase initialized successfully');
} catch (error) {
  console.error('Error initializing Firebase:', error);
  // You might want to show a user-friendly error message here
}

export { db, storage }; 

