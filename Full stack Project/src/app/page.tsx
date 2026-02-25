'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import Sidebar from '@/components/Sidebar';
import { SendIcon, BotIcon, UserIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const starterPrompts = [
    'Summarize this chapter in 6 bullet points.',
    'Explain this topic in simple class-10 language.',
    'Create 5 exam-style questions from this chapter.',
    'What are the important formulas from this section?',
  ];

  const handleSend = async (messageText: string) => {
    if (!messageText.trim()) return;

    // Add user message to UI
    const userMessage = { role: 'user', content: messageText };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [...messages, userMessage] }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail = data?.detail ?? data?.error ?? 'Failed to fetch response';
        throw new Error(detail);
      }

      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }]);
    } catch (error) {
      console.error(error);
      const errorMessage =
        error instanceof Error ? error.message : 'An error occurred while fetching the response.';
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${errorMessage}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#212121] text-[#ececec] font-sans overflow-hidden">
      {/* Sidebar component */}
      <Sidebar />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col items-center justify-between relative h-full">
        {/* Top Header */}
        <div className="w-full max-w-4xl p-4 md:p-6 pb-2 shrink-0">
          <h1 className="text-xl md:text-2xl font-bold tracking-wide">NCERT AI Tutor</h1>
          <p className="text-sm text-[#a8a8a8] mt-1">Focused chat for class 10 NCERT textbooks.</p>
        </div>

        {/* Chat Messages */}
        <ScrollArea className="flex-1 w-full max-w-4xl p-4 md:p-6 my-2">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full pt-10">
              <p className="text-[#a8a8a8] text-sm mb-4">Start with one prompt:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
                {starterPrompts.map((prompt, idx) => (
                  <Button
                    key={idx}
                    variant="outline"
                    className="h-auto py-3 px-4 justify-start text-left border-[#363636] bg-[#242424] hover:bg-[#2d2d2d] hover:text-[#ececec] text-[#ececec] font-normal whitespace-normal w-full"
                    onClick={() => handleSend(prompt)}
                  >
                    {prompt}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col space-y-6 pb-4">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-[#10a37f] flex items-center justify-center shrink-0 mt-1">
                      <BotIcon size={18} className="text-white" />
                    </div>
                  )}

                  <div
                    className={`px-5 py-4 rounded-2xl max-w-[85%] sm:max-w-[75%] border border-[#363636] prose prose-invert prose-p:leading-relaxed ${msg.role === 'assistant'
                        ? 'bg-[#242424] text-[#ececec] rounded-tl-sm'
                        : 'bg-transparent text-[#ececec] rounded-tr-sm'
                      }`}
                  >
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>

                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-full bg-[#3b3b3b] flex items-center justify-center shrink-0 mt-1">
                      <UserIcon size={18} className="text-white" />
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="flex items-center gap-4 animate-pulse pt-2">
                  <div className="w-8 h-8 rounded-full bg-[#10a37f] flex items-center justify-center shrink-0">
                    <BotIcon size={18} className="text-white" />
                  </div>
                  <div className="px-5 py-4 rounded-2xl bg-[#242424] border border-[#363636] text-[#a8a8a8]">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          )}
        </ScrollArea>

        {/* Input Area */}
        <div className="w-full max-w-4xl p-4 md:px-6 md:pb-6 pt-2 shrink-0 bg-gradient-to-t from-[#212121] 30% to-transparent z-10 px-4 pb-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend(input);
            }}
            className="relative flex items-center w-full"
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Message NCERT Tutor..."
              disabled={isLoading}
              className="w-full bg-[#2b2b2b] border-[#363636] text-white rounded-[24px] pl-5 pr-12 py-6 focus-visible:ring-1 focus-visible:ring-[#4a4a4a] text-[15px] hover:border-[#4a4a4a] transition-colors"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              className="absolute right-2 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full bg-white text-black hover:bg-gray-200 disabled:bg-[#3b3b3b] disabled:text-[#a8a8a8] transition-colors"
            >
              <SendIcon size={18} className="ml-0.5" />
            </Button>
          </form>
          <div className="text-center mt-3 text-xs text-[#a8a8a8]">
            AI can make mistakes. Check important info.
          </div>
        </div>
      </main>
    </div>
  );
}
