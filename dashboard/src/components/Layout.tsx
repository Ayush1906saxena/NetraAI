import { Link, Outlet, useLocation } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Screening", icon: "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" },
  { to: "/dashboard", label: "History", icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" },
  { to: "/analytics", label: "Analytics", icon: "M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" },
];

export default function Layout() {
  const { pathname } = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navbar */}
      <header className="bg-gradient-to-r from-[#1B4F72] to-[#2E86C1] text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-3 group">
            {/* Eye Logo */}
            <div className="relative">
              <svg className="w-9 h-9" viewBox="0 0 36 36" fill="none">
                <ellipse cx="18" cy="18" rx="16" ry="11" stroke="white" strokeWidth="1.5" opacity="0.9"/>
                <circle cx="18" cy="18" r="7" stroke="white" strokeWidth="1.5"/>
                <circle cx="18" cy="18" r="3.5" fill="white"/>
                <circle cx="18" cy="18" r="1.5" fill="#1B4F72"/>
                <circle cx="19.5" cy="16.5" r="1" fill="#1B4F72" opacity="0.3"/>
              </svg>
            </div>
            <div>
              <span className="text-xl font-bold tracking-tight">Netra AI</span>
              <span className="hidden sm:block text-[10px] text-white/60 -mt-1 tracking-widest uppercase">Retinal Intelligence</span>
            </div>
          </Link>

          <nav className="flex gap-1">
            {NAV_ITEMS.map(({ to, label, icon }) => (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                  pathname === to
                    ? "bg-white/20 text-white shadow-sm"
                    : "text-white/70 hover:text-white hover:bg-white/10"
                }`}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={icon} />
                </svg>
                <span className="hidden sm:inline">{label}</span>
              </Link>
            ))}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="fade-in">
          <Outlet />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200/50 py-6 text-center">
        <p className="text-sm text-gray-400">
          <span className="font-medium text-gray-500">Netra AI</span> &mdash; AI-Powered Diabetic Retinopathy Screening
        </p>
        <p className="text-xs text-gray-300 mt-1">
          Not a clinical diagnosis. Always consult an ophthalmologist.
        </p>
      </footer>
    </div>
  );
}
