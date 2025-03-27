import VizGallery from '../components/VizGallery';

const Results = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Data Analysis Results
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-xl text-gray-500 sm:mt-4">
            Explore the AI-generated visualizations and insights from your dataset
          </p>
        </div>

        <div className="mt-12">
          <VizGallery />
        </div>
      </div>
    </div>
  );
};

export default Results; 