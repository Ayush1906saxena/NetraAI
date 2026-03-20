import { Link } from "react-router-dom";
import { useEffect, useRef, useState } from "react";

/* ── Scroll-reveal hook ───────────────────────────────────────────────── */

function useReveal() {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return { ref, className: visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8", style: { transition: "opacity 0.7s ease, transform 0.7s ease" } };
}

/* ── Animated counter ─────────────────────────────────────────────────── */

function AnimatedStat({ label, value, suffix = "" }: { label: string; value: string; suffix?: string }) {
  const reveal = useReveal();
  return (
    <div ref={reveal.ref} className={reveal.className} style={reveal.style}>
      <div className="text-center">
        <p className="text-3xl sm:text-4xl font-extrabold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          {value}{suffix}
        </p>
        <p className="text-sm text-gray-500 mt-1 font-medium">{label}</p>
      </div>
    </div>
  );
}

/* ── Floating shapes for hero ─────────────────────────────────────────── */

function HeroBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {/* Mesh gradient */}
      <div className="absolute inset-0 gradient-mesh" />
      {/* Floating orbs */}
      <div
        className="absolute w-72 h-72 rounded-full opacity-[0.07]"
        style={{
          background: "radial-gradient(circle, #6366f1 0%, transparent 70%)",
          top: "10%", left: "5%",
          animation: "float 6s ease-in-out infinite",
        }}
      />
      <div
        className="absolute w-96 h-96 rounded-full opacity-[0.05]"
        style={{
          background: "radial-gradient(circle, #8b5cf6 0%, transparent 70%)",
          top: "20%", right: "-5%",
          animation: "float 8s ease-in-out infinite 1s",
        }}
      />
      <div
        className="absolute w-64 h-64 rounded-full opacity-[0.06]"
        style={{
          background: "radial-gradient(circle, #a78bfa 0%, transparent 70%)",
          bottom: "10%", left: "30%",
          animation: "float 7s ease-in-out infinite 2s",
        }}
      />
      {/* Grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: "radial-gradient(circle, #6366f1 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />
    </div>
  );
}

/* ── Step card ────────────────────────────────────────────────────────── */

function StepCard({ step, title, desc, icon, delay }: { step: number; title: string; desc: string; icon: React.ReactNode; delay: number }) {
  const reveal = useReveal();
  return (
    <div ref={reveal.ref} className={reveal.className} style={{ ...reveal.style, transitionDelay: `${delay}ms` }}>
      <div className="card-elevated p-8 text-center h-full hover:scale-[1.02] transition-transform duration-300">
        <div className="w-14 h-14 rounded-2xl gradient-brand flex items-center justify-center mx-auto mb-5 shadow-[0_4px_16px_rgba(99,102,241,0.25)]">
          {icon}
        </div>
        <span className="inline-block px-3 py-1 rounded-full bg-indigo-50 text-indigo-600 text-xs font-bold mb-3">
          Step {step}
        </span>
        <h3 className="text-lg font-bold text-gray-900 mb-2">{title}</h3>
        <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

/* ── Feature card ─────────────────────────────────────────────────────── */

function FeatureCard({ title, desc, icon, delay }: { title: string; desc: string; icon: React.ReactNode; delay: number }) {
  const reveal = useReveal();
  return (
    <div ref={reveal.ref} className={reveal.className} style={{ ...reveal.style, transitionDelay: `${delay}ms` }}>
      <div className="card p-6 h-full hover:scale-[1.02] transition-transform duration-300 group">
        <div className="w-11 h-11 rounded-xl bg-indigo-50 group-hover:bg-indigo-100 flex items-center justify-center mb-4 transition-colors duration-300">
          {icon}
        </div>
        <h3 className="text-base font-bold text-gray-900 mb-1.5">{title}</h3>
        <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}

/* ── Tech badge ───────────────────────────────────────────────────────── */

function TechBadge({ name, delay }: { name: string; delay: number }) {
  const reveal = useReveal();
  return (
    <div ref={reveal.ref} className={reveal.className} style={{ ...reveal.style, transitionDelay: `${delay}ms` }}>
      <div className="px-5 py-3 rounded-xl bg-white border border-gray-100 shadow-sm hover:shadow-md hover:border-indigo-100 transition-all duration-300 text-sm font-semibold text-gray-700">
        {name}
      </div>
    </div>
  );
}

/* ── Main landing page ────────────────────────────────────────────────── */

export default function LandingPage() {

  return (
    <div className="min-h-screen bg-[#fafbfc]">

      {/* ── Navbar ──────────────────────────────────────────────────────── */}
      <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-gray-200/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl gradient-brand flex items-center justify-center shadow-[0_2px_8px_rgba(99,102,241,0.3)] group-hover:shadow-[0_4px_16px_rgba(99,102,241,0.4)] transition-shadow duration-300">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight text-gray-900">
                Netra<span className="text-indigo-600">AI</span>
              </span>
              <span className="hidden sm:block text-[10px] text-gray-400 -mt-0.5 tracking-widest uppercase font-medium">
                Retinal Intelligence
              </span>
            </div>
          </Link>
          <nav className="flex items-center gap-2 sm:gap-4">
            <a href="#features" className="hidden sm:inline text-sm text-gray-500 hover:text-gray-900 font-medium transition-colors">Features</a>
            <a href="#validation" className="hidden sm:inline text-sm text-gray-500 hover:text-gray-900 font-medium transition-colors">Validation</a>
            <Link to="/login" className="text-sm text-gray-500 hover:text-gray-900 font-medium transition-colors">Login</Link>
            <Link
              to="/demo"
              className="px-5 py-2 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold shadow-[0_2px_8px_rgba(99,102,241,0.3)] hover:shadow-[0_4px_16px_rgba(99,102,241,0.4)] hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
            >
              Try Demo
            </Link>
          </nav>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────────────────── */}
      <section className="relative pt-32 pb-20 sm:pt-40 sm:pb-28 overflow-hidden">
        <HeroBackground />
        <div className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="fade-in-up">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-600 text-xs font-semibold mb-6 tracking-wide uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
              Clinical-Grade AI Screening
            </div>
          </div>

          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold text-gray-900 tracking-tight leading-[1.1] fade-in-up stagger-1">
            AI-Powered Diabetic
            <br />
            <span className="bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              Retinopathy Screening
            </span>
          </h1>

          <p className="mt-6 text-lg sm:text-xl text-gray-500 max-w-2xl mx-auto leading-relaxed fade-in-up stagger-2">
            Detect diabetic eye disease in seconds. Clinical-grade AI screening
            accessible to everyone.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4 fade-in-up stagger-3">
            <Link
              to="/demo"
              className="group inline-flex items-center gap-2.5 px-8 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-base shadow-[0_4px_20px_rgba(99,102,241,0.35)] hover:shadow-[0_8px_32px_rgba(99,102,241,0.45)] hover:scale-[1.03] active:scale-[0.98] transition-all duration-200"
            >
              Try Demo
              <svg className="w-5 h-5 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <a
              href="#how-it-works"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl border border-gray-200 bg-white text-gray-700 font-semibold text-base hover:border-indigo-200 hover:bg-indigo-50/30 transition-all duration-200"
            >
              Learn More
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </a>
          </div>
        </div>
      </section>

      {/* ── Stats bar ───────────────────────────────────────────────────── */}
      <section className="py-12 border-y border-gray-100 bg-white/60">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
            <AnimatedStat label="Quadratic Weighted Kappa" value="0.892" />
            <AnimatedStat label="Area Under Curve" value="97.6" suffix="%" />
            <AnimatedStat label="Sensitivity" value="90.6" suffix="%" />
            <AnimatedStat label="Analysis Time" value="3s" />
          </div>
        </div>
      </section>

      {/* ── How it works ────────────────────────────────────────────────── */}
      <section id="how-it-works" className="py-20 sm:py-28">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <p className="text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3">How It Works</p>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              Three simple steps
            </h2>
            <p className="mt-4 text-gray-500 max-w-xl mx-auto">
              From fundus image upload to actionable clinical insights in seconds.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <StepCard
              step={1} title="Upload" delay={0}
              desc="Upload a fundus photograph from any standard retinal camera. Supports JPEG and PNG formats."
              icon={
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
              }
            />
            <StepCard
              step={2} title="Analyze" delay={100}
              desc="Our ONNX-optimized deep learning model processes the image with GradCAM explainability in real time."
              icon={
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.59.659H9.06a2.25 2.25 0 01-1.591-.659L5 14.5m14 0V18a2.25 2.25 0 01-2.25 2.25H7.25A2.25 2.25 0 015 18v-3.5" />
                </svg>
              }
            />
            <StepCard
              step={3} title="Results" delay={200}
              desc="Receive a 5-class DR grading with confidence scores, GradCAM heatmap, and a downloadable PDF report."
              icon={
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15a2.25 2.25 0 012.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25" />
                </svg>
              }
            />
          </div>
        </div>
      </section>

      {/* ── Features ────────────────────────────────────────────────────── */}
      <section id="features" className="py-20 sm:py-28 bg-gray-50/60">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <p className="text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3">Features</p>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              Everything you need for retinal screening
            </h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard
              delay={0}
              title="5-Class DR Grading"
              desc="From No DR to Proliferative DR -- comprehensive severity classification aligned with international standards."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" /></svg>}
            />
            <FeatureCard
              delay={80}
              title="GradCAM Explainability"
              desc="See exactly what the AI sees. Gradient-weighted class activation maps highlight the regions driving each prediction."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
            />
            <FeatureCard
              delay={160}
              title="On-Device Quality Check"
              desc="Real-time image quality guidance ensures only diagnostic-grade images are processed by the AI model."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
            />
            <FeatureCard
              delay={240}
              title="Instant PDF Reports"
              desc="Generate patient-friendly PDF reports with QR codes for easy sharing with referring physicians."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>}
            />
            <FeatureCard
              delay={320}
              title="Glaucoma Screening"
              desc="Automated cup-to-disc ratio analysis for glaucoma risk assessment alongside diabetic retinopathy grading."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
            />
            <FeatureCard
              delay={400}
              title="63x Faster with ONNX"
              desc="Production-optimized ONNX Runtime inference delivers sub-second predictions without sacrificing accuracy."
              icon={<svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" /></svg>}
            />
          </div>
        </div>
      </section>

      {/* ── Clinical Validation ──────────────────────────────────────────── */}
      <section id="validation" className="py-20 sm:py-28">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <p className="text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3">Clinical Validation</p>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight">
              Validated across multiple datasets
            </h2>
            <p className="mt-4 text-gray-500 max-w-xl mx-auto">
              Rigorous evaluation on both in-distribution and cross-dataset benchmarks ensures real-world reliability.
            </p>
          </div>
          <ValidationTable />
        </div>
      </section>

      {/* ── Tech Stack ──────────────────────────────────────────────────── */}
      <section className="py-20 sm:py-28 bg-gray-50/60">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3">Tech Stack</p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight mb-4">
            Built with modern tools
          </h2>
          <p className="text-gray-500 max-w-xl mx-auto mb-12">
            End-to-end deep learning pipeline from training to production, with a modern full-stack web and mobile experience.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            {["PyTorch", "FastAPI", "React", "Flutter", "PostgreSQL", "Redis", "ONNX Runtime", "Docker"].map((name, i) => (
              <TechBadge key={name} name={name} delay={i * 60} />
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────────── */}
      <section className="py-20 sm:py-28">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="card-elevated p-10 sm:p-16 relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute inset-0 gradient-mesh opacity-60" />
            <div className="absolute -top-24 -right-24 w-64 h-64 rounded-full bg-indigo-100/40" />
            <div className="absolute -bottom-16 -left-16 w-48 h-48 rounded-full bg-purple-100/30" />

            <div className="relative z-10">
              <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 tracking-tight mb-4">
                Ready to screen?
              </h2>
              <p className="text-gray-500 max-w-lg mx-auto mb-8 leading-relaxed">
                Upload a fundus image and get an AI-powered diabetic retinopathy screening result in seconds -- completely free.
              </p>
              <Link
                to="/demo"
                className="group inline-flex items-center gap-2.5 px-10 py-4 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold text-lg shadow-[0_4px_20px_rgba(99,102,241,0.35)] hover:shadow-[0_8px_32px_rgba(99,102,241,0.45)] hover:scale-[1.03] active:scale-[0.98] transition-all duration-200"
              >
                Try Demo Now
                <svg className="w-5 h-5 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <p className="mt-6 text-xs text-gray-400 max-w-md mx-auto">
                This is an AI-assisted screening tool and does not constitute a clinical diagnosis.
                Always consult a qualified ophthalmologist for medical decisions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <footer className="border-t border-gray-100 py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
                <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div>
                <span className="text-base font-bold text-gray-900">
                  Netra<span className="text-indigo-600">AI</span>
                </span>
                <p className="text-xs text-gray-400">AI-Powered Retinal Intelligence</p>
              </div>
            </div>

            <nav className="flex items-center gap-6 text-sm text-gray-500">
              <Link to="/demo" className="hover:text-gray-900 transition-colors">Demo</Link>
              <Link to="/dashboard" className="hover:text-gray-900 transition-colors">Dashboard</Link>
              <Link to="/analytics" className="hover:text-gray-900 transition-colors">Analytics</Link>
              <Link to="/login" className="hover:text-gray-900 transition-colors">Login</Link>
            </nav>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-100 text-center">
            <p className="text-xs text-gray-400">
              Not a clinical diagnosis. This AI screening tool is intended for research and education purposes.
              Always consult a qualified ophthalmologist for clinical decision-making.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

/* ── Validation table component ───────────────────────────────────────── */

function ValidationTable() {
  const reveal = useReveal();

  const metrics = [
    { name: "Quadratic Weighted Kappa", aptos: "0.892", idrid: "0.841" },
    { name: "AUC (macro-avg)", aptos: "97.6%", idrid: "94.2%" },
    { name: "Sensitivity", aptos: "90.6%", idrid: "87.3%" },
    { name: "Specificity", aptos: "96.8%", idrid: "95.1%" },
  ];

  return (
    <div ref={reveal.ref} className={reveal.className} style={reveal.style}>
      <div className="card-elevated overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-4 px-6 text-gray-500 font-semibold">Metric</th>
                <th className="text-center py-4 px-6 text-gray-500 font-semibold">
                  <span className="inline-flex items-center gap-2">
                    APTOS 2019
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 font-semibold">In-Distribution</span>
                  </span>
                </th>
                <th className="text-center py-4 px-6 text-gray-500 font-semibold">
                  <span className="inline-flex items-center gap-2">
                    IDRiD
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-50 text-purple-600 font-semibold">Cross-Dataset</span>
                  </span>
                </th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, i) => (
                <tr key={m.name} className={i < metrics.length - 1 ? "border-b border-gray-50" : ""}>
                  <td className="py-4 px-6 font-medium text-gray-800">{m.name}</td>
                  <td className="py-4 px-6 text-center">
                    <span className="font-semibold text-indigo-600">{m.aptos}</span>
                  </td>
                  <td className="py-4 px-6 text-center">
                    <span className="font-semibold text-purple-600">{m.idrid}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
