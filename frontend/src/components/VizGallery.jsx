import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const VizGallery = () => {
  const [visualizations, setVisualizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    try {
      const data = JSON.parse(localStorage.getItem('visualizationData'));
      if (data && data.visualizations) {
        setVisualizations(data.visualizations);
      } else {
        setError('No visualization data found');
      }
    } catch (err) {
      setError('Error loading visualization data');
    } finally {
      setLoading(false);
    }
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center p-6">
        <div className="text-red-600 mb-4">{error}</div>
        <p className="text-gray-600">
          Please upload a dataset to view visualizations
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 p-6">
      {visualizations.map((viz, index) => (
        <div
          key={index}
          className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow"
        >
          <Plot
            data={JSON.parse(viz.data).data}
            layout={{
              ...JSON.parse(viz.data).layout,
              height: 400,
              margin: { t: 30, b: 30, l: 30, r: 30 },
            }}
            config={{ responsive: true }}
            className="w-full h-full"
          />
        </div>
      ))}
    </div>
  );
};

export default VizGallery; 