import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Store the response data in localStorage
      localStorage.setItem('visualizationData', JSON.stringify(response.data));
      navigate('/results');
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while uploading the file');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Dataset</h2>
        <form onSubmit={handleSubmit}>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-500'}`}
          >
            <input {...getInputProps()} />
            {file ? (
              <div>
                <p className="text-sm text-gray-600">Selected file:</p>
                <p className="text-sm font-medium text-gray-900">{file.name}</p>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                  }}
                  className="mt-2 text-sm text-red-600 hover:text-red-800"
                >
                  Remove file
                </button>
              </div>
            ) : (
              <div>
                <p className="text-gray-600">
                  {isDragActive
                    ? 'Drop the file here'
                    : 'Drag and drop a file here, or click to select'}
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Supported formats: CSV, Excel (.xlsx, .xls)
                </p>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-md">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !file}
            className={`mt-6 w-full py-2 px-4 rounded-md text-white font-medium
              ${loading || !file
                ? 'bg-primary-400 cursor-not-allowed'
                : 'bg-primary-600 hover:bg-primary-700'}`}
          >
            {loading ? 'Processing...' : 'Upload and Analyze'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default UploadForm; 