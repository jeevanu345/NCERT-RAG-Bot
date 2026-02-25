import { put } from '@vercel/blob';
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    try {
        // 1. Get filename from query
        const { searchParams } = new URL(request.url);
        const filename = searchParams.get('filename');

        if (!filename) {
            return NextResponse.json({ error: 'Filename is required' }, { status: 400 });
        }

        // 2. Upload directly to Vercel Blob
        const blob = await put(filename, request.body as ReadableStream, {
            access: 'public',
        });

        // 3. Trigger Python backend to process the file and insert into Pinecone
        try {
            // In production, this targets the python backend via the rewrite rules
            // Note: we don't await this if we want it to be asynchronous, but for simplicity we'll await it 
            // since Vercel serverless has a timeout. To avoid timeout, we might need a background queue,
            // but for Vercel Hobby we will await it or process it.

            const origin = new URL(request.url).origin;
            const baseURL = process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : origin;
            const pythonResponse = await fetch(`${baseURL}/api/process_pdf`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    file_url: blob.url,
                    filename: filename
                })
            });

            if (!pythonResponse.ok) {
                const errData = await pythonResponse.text();
                console.error("Python processing error:", errData);
                // We still return the blob so the user knows upload worked, even if processing failed
                return NextResponse.json({ ...blob, processingError: "File uploaded but failed to vectorize" });
            }

            const pythonData = await pythonResponse.json();
            return NextResponse.json({ ...blob, ...pythonData });

        } catch (apiError) {
            console.error("Error triggering python backend:", apiError);
            return NextResponse.json({ ...blob, processingError: "Backend processing unreachable" });
        }

    } catch (error) {
        console.error('Upload Error:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
