import { db } from '../firebase';
import { doc, setDoc, getDoc, collection, query, orderBy, limit, getDocs } from '@firebase/firestore';

const handleFirebaseError = (error, context) => {
  console.error(`Firebase error during ${context}:`, error);
  throw new Error(`Firebase error during ${context}: ${error.message}`);
};

const checkFirebaseConnection = () => {
  if (!db) {
    throw new Error('Firebase is not initialized. Please check your configuration.');
  }
};

export const firebaseService = {
  // Store dataset and its metadata
  async storeDataset(file, metadata) {
    try {
      const timestamp = Date.now();
      const docRef = doc(db, 'datasets', timestamp.toString());
      await setDoc(docRef, {
        ...metadata,
        uploadTime: timestamp
      });
      return timestamp.toString();
    } catch (error) {
      handleFirebaseError(error, 'storing dataset');
    }
  },

  // Store cleaned dataset
  async storeCleanedDataset(datasetId, cleaningData) {
    try {
      const docRef = doc(db, 'cleaned_datasets', datasetId);
      await setDoc(docRef, cleaningData);
    } catch (error) {
      handleFirebaseError(error, 'storing cleaned dataset');
    }
  },

  // Store feature rankings
  async storeFeatureRankings(datasetId, rankings) {
    try {
      const docRef = doc(db, 'feature_rankings', datasetId);
      await setDoc(docRef, rankings);
    } catch (error) {
      handleFirebaseError(error, 'storing feature rankings');
    }
  },

  // Store visualizations
  async storeVisualizations(datasetId, visualizations) {
    try {
      const docRef = doc(db, 'datasets', datasetId);
      const visualizationData = {
        summary: visualizations.summary || {},
        plots: visualizations.plots || {},
        timestamp: visualizations.timestamp || Date.now()
      };
      
      // Convert any non-serializable data to serializable format
      const serializedData = {
        summary: JSON.parse(JSON.stringify(visualizationData.summary)),
        plots: JSON.parse(JSON.stringify(visualizationData.plots)),
        timestamp: visualizationData.timestamp
      };
      
      await setDoc(docRef, { visualizations: serializedData }, { merge: true });
    } catch (error) {
      console.error('Error storing visualizations:', error);
      throw error;
    }
  },

  // Get dataset details
  async getDatasetDetails(datasetId) {
    try {
      checkFirebaseConnection();
      const docRef = doc(db, 'datasets', datasetId);
      const docSnap = await getDoc(docRef);
      if (!docSnap.exists()) {
        throw new Error('Dataset not found');
      }
      return docSnap.data();
    } catch (error) {
      handleFirebaseError(error, 'getting dataset details');
    }
  },

  // Get cleaned dataset
  async getCleanedDataset(datasetId) {
    try {
      const docRef = doc(db, 'cleaned_datasets', datasetId);
      const docSnap = await getDoc(docRef);
      if (!docSnap.exists()) {
        throw new Error('Cleaned dataset not found');
      }
      return docSnap.data();
    } catch (error) {
      handleFirebaseError(error, 'getting cleaned dataset');
    }
  },

  // Get feature rankings
  async getFeatureRankings(datasetId) {
    try {
      const docRef = doc(db, 'feature_rankings', datasetId);
      const docSnap = await getDoc(docRef);
      if (!docSnap.exists()) {
        throw new Error('Feature rankings not found');
      }
      return docSnap.data();
    } catch (error) {
      handleFirebaseError(error, 'getting feature rankings');
    }
  },

  // Get visualizations
  async getVisualizations(datasetId) {
    try {
      console.log('Getting visualizations for dataset:', datasetId);
      const docRef = doc(db, 'datasets', datasetId);
      const docSnap = await getDoc(docRef);
      console.log('Visualization data from Firebase:', docSnap.data());
      
      if (!docSnap.exists()) {
        console.log('No visualizations found in Firebase');
        return { summary: {}, plots: {}, timestamp: null };
      }
      
      const data = docSnap.data();
      if (!data.visualizations) {
        console.log('No visualizations found in dataset');
        return { summary: {}, plots: {}, timestamp: null };
      }
      
      return data.visualizations;
    } catch (error) {
      console.error('Error getting visualizations:', error);
      handleFirebaseError(error, 'getting visualizations');
    }
  },

  // Get recent datasets
  async getRecentDatasets(limit = 5) {
    try {
      checkFirebaseConnection();
      const q = query(collection(db, 'datasets'), orderBy('uploadTime', 'desc'), limit(limit));
      const querySnapshot = await getDocs(q);
      return querySnapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
    } catch (error) {
      handleFirebaseError(error, 'getting recent datasets');
    }
  }
}; 