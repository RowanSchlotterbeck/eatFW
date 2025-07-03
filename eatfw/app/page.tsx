"use client";

import { useState, useRef, useEffect } from "react";

// Define the type message, makes it easier for the Message type decleration
interface Restaurant {
  name: string;
  shortDescription: string;
  address?: string;
  matchHighlights?: string;
}

export default function Home() {
  const [restaurants, setRestaurants] = useState<Restaurant[]>([]); // Holds the 3-card results
  const [isLoading, setIsLoading] = useState(false); // State that tracks the loading state during an API call
  const [lastQuery, setLastQuery] = useState(""); // Remembers the user's last submitted query
  const [inputValue, setInputValue] = useState(""); // State that tracks the input of the user, will be effected by STT if used
  const [isListening, setIsListening] = useState(false); // State the tracks if the client is listening to the user
  const [pastQueries, setPastQueries] = useState<string[]>([]); // Stores recent user queries for chip history
  const recognitionRef = useRef<any>(null); // Used to define the recognition object, must be type any and defualted to null

  // Create the Speech Recogntion object in a UseEffect hook
  // This is where the website asks the user if it can record its voice

  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn("Speech recognition not supported in this browser.");
      // Here you could disable the microphone button
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true; // Listens until the users hits the button again
    recognition.interimResults = true; // The magic line for live transcripitions
    recognition.lang = "en-US"; // Defines the language

    // On start, the setIsListening state will be set to true
    recognition.onstart = () => {
      setIsListening(true);
    };

    // On end, the setIsListening state will be set to false
    recognition.onend = () => {
      setIsListening(false);
    };

    // Reports an error if one is thrown
    recognition.onerror = (e: any) => {
      console.error("Speech recognition error:", e.error);
      setIsListening(false);
    };

    // Updates the input fields value as the user continues to talk to the app
    recognition.onresult = (e: any) => {
      const transcript = Array.from(e.results)
        .map((result: any) => result[0])
        .map((result: any) => result.transcript)
        .join("");
      setInputValue(transcript);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
    };
  }, []);

  // Listening Logic for Web Speech API
  // Only does recognition when not listening
  const handleListen = () => {
    const recognition = recognitionRef.current;
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
    }
  };

  // Helper to fetch restaurants based on a given query
  const fetchRestaurants = async (query: string) => {
    setLastQuery(query);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRestaurants(data.restaurants ?? []);
    } catch (error) {
      console.error("Failed to fetch from the API:", error);
      setRestaurants([]);
    } finally {
      setIsLoading(false);
    }
  };

  // When the user submits, a message is appended to the messages array -> API is called
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    const userQuery = inputValue.trim();
    setInputValue("");

    // Update chip history: ensure uniqueness & cap at 10 entries
    setPastQueries((prev) => {
      const updated = [userQuery, ...prev.filter((q) => q !== userQuery)];
      return updated.slice(0, 10);
    });

    await fetchRestaurants(userQuery);
  };

  // Handle clicking a query chip
  const handleChipClick = (query: string) => {
    fetchRestaurants(query);
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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Informational banner about the current query */}
          {!isLoading && lastQuery && restaurants.length > 0 && (
            <div className="col-span-full text-center text-sm text-gray-500 dark:text-gray-400 mb-2">
              Showing results for <span className="italic">"{lastQuery}"</span>
            </div>
          )}

          {restaurants.length === 0 && !isLoading ? (
            <div className="col-span-full flex items-center justify-center text-gray-400">
              <div className="text-center">
                <p>Ask a question to get started!</p>
                <p className="text-xs">
                  e.g., "Where can I find the best BBQ?"
                </p>
              </div>
            </div>
          ) : (
            restaurants.map((restaurant, index) => (
              <div
                key={index}
                className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 flex flex-col"
              >
                <h2 className="text-xl font-semibold mb-2">
                  {restaurant.name}
                </h2>
                <p className="text-gray-700 dark:text-gray-300 flex-1">
                  {restaurant.shortDescription}
                </p>
                {restaurant.address && (
                  <p className="text-sm text-gray-500 mt-4">
                    {restaurant.address}
                  </p>
                )}
              </div>
            ))
          )}
          {isLoading && (
            <div className="col-span-full flex justify-center">
              <p className="animate-pulse">
                Searching Fort Worth for "{lastQuery}" …
              </p>
            </div>
          )}
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
                onClick={handleListen}
                className={`p-2 text-gray-500 rounded-full hover:bg-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 cursor-pointer ${
                  isListening ? "bg-red-500 text-white animate-pulse" : ""
                }`}
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

          {/* History chips */}
          {pastQueries.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3 overflow-x-auto">
              {pastQueries.map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleChipClick(query)}
                  className="px-3 py-1 mt-5 rounded-full bg-gray-200 dark:bg-gray-700 text-sm hover:bg-gray-300 dark:hover:bg-gray-600 whitespace-nowrap"
                >
                  {query}
                </button>
              ))}
            </div>
          )}
        </div>
      </footer>
    </div>
  );
}
