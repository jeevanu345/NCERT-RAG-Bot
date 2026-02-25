'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { PlusIcon, UploadIcon, FileTextIcon, SettingsIcon } from 'lucide-react';

export default function Sidebar() {
    const [sources, setSources] = useState(['ncert_text.txt']);
    const [isUploading, setIsUploading] = useState(false);

    // Call our Next.js API route
    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        try {
            const response = await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
                method: 'POST',
                body: file,
            });

            if (!response.ok) {
                throw new Error('Upload failed');
            }

            const data = await response.json();
            console.log('Upload success:', data);

            // Add the file to the sources list if it's not already there
            if (!sources.includes(file.name)) {
                setSources((prev) => [file.name, ...prev]);
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Failed to upload and process PDF');
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <aside className="w-64 md:w-72 bg-[#171717] border-r border-[#363636] flex flex-col h-full hidden md:flex shrink-0">
            <div className="p-4 flex flex-col gap-4">
                {/* Brand */}
                <div className="font-semibold text-[1.05rem] tracking-wide text-[#ececec] mb-2 px-1">
                    NCERT Tutor
                </div>

                {/* New Chat Button */}
                <Button
                    variant="outline"
                    className="w-full justify-start gap-2 border-[#363636] bg-transparent text-[#ececec] hover:bg-[#2b2b2b] hover:text-white"
                    onClick={() => window.location.reload()}
                >
                    <PlusIcon size={16} />
                    New chat
                </Button>
            </div>

            {/* Textbook Source Section */}
            <div className="px-4 py-2 mt-2">
                <h3 className="text-[0.75rem] font-semibold text-[#a8a8a8] uppercase tracking-wider mb-3 px-1">
                    Textbook Source
                </h3>

                {/* Upload Input */}
                <div className="mb-4">
                    <label
                        htmlFor="pdf-upload"
                        className={`flex items-center gap-2 w-full p-2.5 rounded-lg border border-dashed border-[#4a4a4a] bg-[#242424] hover:bg-[#2b2b2b] text-sm text-[#ececec] cursor-pointer transition-colors ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
                    >
                        <UploadIcon size={16} className="text-[#a8a8a8]" />
                        {isUploading ? 'Uploading...' : 'Upload NCERT PDF'}
                        <input
                            id="pdf-upload"
                            type="file"
                            accept="application/pdf"
                            className="hidden"
                            onChange={handleUpload}
                        />
                    </label>
                </div>

                {/* File List */}
                <div className="flex flex-col gap-1">
                    {sources.map((src, i) => (
                        <button
                            key={i}
                            className="flex items-center gap-2 w-full p-2 rounded-lg bg-[#2b2b2b] text-sm text-[#ececec] transition-colors text-left"
                        >
                            <FileTextIcon size={16} className="text-[#10a37f] shrink-0" />
                            <span className="truncate">{src}</span>
                        </button>
                    ))}
                </div>
            </div>

            <div className="mt-auto p-4 border-t border-[#363636]">
                <Button variant="ghost" className="w-full justify-start gap-2 text-[#a8a8a8] hover:text-[#ececec] hover:bg-[#2b2b2b]">
                    <SettingsIcon size={16} />
                    Settings
                </Button>
            </div>
        </aside>
    );
}
