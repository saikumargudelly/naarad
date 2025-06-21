import React from 'react';
import { motion } from 'framer-motion';
import { 
  HomeIcon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  UserCircleIcon,
  BellIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';

const Sidebar = ({ 
  sidebarOpen, 
  setSidebarOpen, 
  activeTab, 
  setActiveTab, 
  darkMode, 
  setDarkMode 
}) => {
  const navigation = [
    { name: 'Dashboard', icon: HomeIcon, id: 'dashboard' },
    { name: 'Chat', icon: ChatBubbleLeftRightIcon, id: 'chat' },
    { name: 'Analytics', icon: ChartBarIcon, id: 'analytics' },
    { name: 'Settings', icon: Cog6ToothIcon, id: 'settings' },
  ];

  const getIcon = (iconName, active = false) => {
    const IconComponent = iconName;
    return (
      <IconComponent 
        className={`w-6 h-6 ${active ? 'text-purple-600' : 'text-gray-600 dark:text-gray-400'}`} 
      />
    );
  };

  return (
    <div className="flex flex-col w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700">
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm font-bold">N</span>
            </div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              Naarad
            </h1>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigation.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <motion.button
                key={item.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg transition-colors ${
                  isActive
                    ? 'bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                {getIcon(item.icon, isActive)}
                <span className="font-medium">{item.name}</span>
              </motion.button>
            );
          })}
        </nav>

        {/* Bottom section */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 space-y-4">
          {/* Search */}
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          {/* User section */}
          <div className="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <UserCircleIcon className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                User
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                user@example.com
              </p>
            </div>
            <button className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
              <BellIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Dark mode toggle */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="w-full flex items-center justify-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <span>{darkMode ? '🌙' : '☀️'}</span>
            <span>{darkMode ? 'Dark Mode' : 'Light Mode'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 