/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#14213d",
        moss: "#0b6e4f",
        sand: "#f4efe4",
        clay: "#c45c26",
        mist: "#d9e5de",
      },
      fontFamily: {
        display: ['"Fraunces"', "Georgia", "serif"],
        sans: ['"Source Sans 3"', "system-ui", "sans-serif"],
      },
      backgroundImage: {
        meadow: "radial-gradient(circle at 20% 20%, #e7f2ea 0%, transparent 45%), radial-gradient(circle at 80% 0%, #f8e8d8 0%, transparent 40%), linear-gradient(160deg, #f4efe4, #e8f0ea)",
      },
    },
  },
  plugins: [],
};
