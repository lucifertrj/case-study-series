/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: "/",
        destination: "/knowledge",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
