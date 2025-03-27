import UploadForm from '../components/UploadForm';

const Upload = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Upload Your Dataset
          </h2>
          <p className="mt-3 max-w-2xl mx-auto text-xl text-gray-500 sm:mt-4">
            Upload your CSV or Excel file to begin the analysis
          </p>
        </div>

        <div className="mt-12">
          <UploadForm />
        </div>
      </div>
    </div>
  );
};

export default Upload; 