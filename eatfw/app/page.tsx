"use client";

import { useState } from "react";

export default function Home() {
  const [inputValue, setInputValue] = useState("");

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim()) return;
    console.log("User Input:", inputValue);
    // Logic to handle submission will go here
    setInputValue("");
  };

  return (
    <div className="flex flex-col h-screen font-sans bg-gray-50 dark:bg-black">
      <header className="py-4 px-6 border-b dark:border-gray-800">
        <h1 className="text-2xl font-bold text-center text-gray-900 dark:text-gray-100">
          EatFW
        </h1>
        <p className="text-sm text-center text-gray-500 dark:text-gray-400">
          Your AI guide for Fort Worth's best food
        </p>
      </header>

      <main className="flex-1 overflow-y-auto p-6">
        <div className="flex flex-col space-y-4">
          {/* Chat messages will be rendered here */}
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p>Ask a question to get started!</p>
              <p className="text-xs">e.g., "Where can I find the best BBQ?"</p>
            </div>
          </div>
        </div>
      </main>

      <footer className="p-4 bg-white dark:bg-gray-900/50 border-t dark:border-gray-800">
        <div className="max-w-2xl mx-auto">
          <form
            onSubmit={handleSubmit}
            className="flex items-center p-1.5 pl-4 bg-gray-100 dark:bg-gray-800 rounded-full"
          >
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="What are you in the mood for?"
              className="flex-1 w-full text-gray-800 bg-transparent dark:text-gray-200 focus:outline-none"
            />
            <div className="flex gap-x-2">
              <button
                type="button"
                className="p-2 text-gray-500 rounded-full hover:bg-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 cursor-pointer"
                // onClick for voice will be implemented later
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="w-5 h-5"
                >
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" x2="12" y1="19" y2="22"></line>
                </svg>
              </button>
              <button
                type="submit"
                className="p-2 text-white bg-blue-600 rounded-full hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed cursor-pointer"
                disabled={!inputValue.trim()}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="w-5 h-5"
                >
                  <path d="m22 2-7 20-4-9-9-4Z" />
                  <path d="M22 2 11 13" />
                </svg>
              </button>
            </div>
          </form>
        </div>
      </footer>
    </div>
  );
}
