/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f4f6fa",
          100: "#e6eaf2",
          200: "#c9d1e1",
          300: "#9eabc6",
          400: "#6f7fa3",
          500: "#4e5f86",
          600: "#3b4a6d",
          700: "#2e3a57",
          800: "#1f2840",
          900: "#141b2e",
          950: "#0b1020",
        },
        sun: {
          50: "#fff6ed",
          100: "#ffead4",
          200: "#ffd0a8",
          300: "#ffae70",
          400: "#ff8136",
          500: "#fb6110",
          600: "#ec4706",
          700: "#c43308",
          800: "#9c2a0f",
          900: "#7e2510",
        },
        moss: {
          400: "#34c39a",
          500: "#14a37c",
          600: "#0b8465",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "PingFang SC",
          "Hiragino Sans GB",
          "Microsoft YaHei",
          "sans-serif",
        ],
        display: ["Outfit", "Inter", "PingFang SC", "Microsoft YaHei", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(20,27,46,0.04), 0 8px 24px -12px rgba(20,27,46,0.12)",
        float: "0 24px 60px -20px rgba(20,27,46,0.35)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "fade-up": "fade-up .35s ease-out both",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
};
