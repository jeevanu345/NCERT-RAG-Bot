/** @type {import('next').NextConfig} */
const nextConfig = {
    // Keep chat on Next route handlers, and only proxy endpoints implemented by FastAPI.
    rewrites: async () => {
        return [
            {
                source: '/api/process_pdf',
                destination:
                    process.env.NODE_ENV === 'development'
                        ? 'http://127.0.0.1:8000/api/process_pdf'
                        : '/api/process_pdf',
            },
        ];
    },
};

module.exports = nextConfig;
