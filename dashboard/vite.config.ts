import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4173,
    // Docker Compose exposes Nginx on :80 as the public API entrypoint.
    // Override with VITE_API_TARGET=http://localhost:8000 when running only Uvicorn.
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://localhost:80",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
