/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070E1D',
          900: '#0D1830',
          800: '#132242',
          700: '#1B2F59',
          600: '#27427A',
        },
        turquoise: {
          DEFAULT: '#59D9C5',
          400: '#76E3D2',
          500: '#59D9C5',
          600: '#3BB8A4',
        },
        violet: {
          DEFAULT: '#986DC7',
          400: '#B08CD7',
          500: '#986DC7',
          600: '#7B4FB0',
        },
        electric: {
          DEFAULT: '#38BDF8',
          500: '#0EA5E9',
        },
        coral: {
          DEFAULT: '#FB923C',
          500: '#F97316',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
