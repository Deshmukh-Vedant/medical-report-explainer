/**
 * Application Runtime Configuration
 *
 * For Netlify Deployment:
 * Set window.RENDER_BACKEND_URL to your deployed Render Web Service URL.
 * Example: "https://your-medical-app.onrender.com"
 *
 * If left empty, main.js will automatically fall back to the default Render URL
 * or the local development server (http://127.0.0.1:8000).
 */
window.RENDER_BACKEND_URL = window.RENDER_BACKEND_URL || "https://medical-report-explainer-backend.onrender.com";
