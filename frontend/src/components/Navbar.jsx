import { Link } from 'react-router-dom';
import { HomeIcon, ChartBarIcon, ArrowUpTrayIcon } from '@heroicons/react/24/outline';

const Navbar = () => {
  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
            <Link
                to="/"
                className="inline-flex items-center px-1 pt-1 text-gray-900 hover:text-primary-600"
              >
              <span className="text-2xl font-bold text-primary-600">DataViz</span>
              </Link>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              <Link
                to="/"
                className="inline-flex items-center px-1 pt-1 text-gray-900 hover:text-primary-600"
              >
                <HomeIcon className="h-5 w-5 mr-1" />
                Home
              </Link>
              <Link
                to="/upload"
                className="inline-flex items-center px-1 pt-1 text-gray-900 hover:text-primary-600"
              >
                <ArrowUpTrayIcon className="h-5 w-5 mr-1" />
                Upload
              </Link>
              <Link
                to="/results"
                className="inline-flex items-center px-1 pt-1 text-gray-900 hover:text-primary-600"
              >
                <ChartBarIcon className="h-5 w-5 mr-1" />
                Results
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 