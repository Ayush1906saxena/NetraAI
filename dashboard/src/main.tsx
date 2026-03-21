import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";
import Layout from "./components/Layout";
import LandingPage from "./pages/LandingPage";
import DemoPage from "./pages/DemoPage";
import DashboardPage from "./pages/DashboardPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import LoginPage from "./pages/LoginPage";
import CompareEyesPage from "./pages/CompareEyesPage";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* Landing page has its own layout (no app chrome) */}
        <Route path="/" element={<LandingPage />} />

        {/* App pages with shared navbar / footer */}
        <Route element={<Layout />}>
          <Route path="/demo" element={<DemoPage />} />
          <Route path="/compare" element={<CompareEyesPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/login" element={<LoginPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
