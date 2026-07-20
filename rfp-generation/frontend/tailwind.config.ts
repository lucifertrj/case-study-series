import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef6ff",
          100: "#d9eaff",
          500: "#2563eb",
          600: "#1d4ed8",
          700: "#1e40af",
        },
        ink: "#141821",
        line: "#d8dee8",
      },
      boxShadow: {
        soft: "0 14px 38px rgba(15, 23, 42, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
