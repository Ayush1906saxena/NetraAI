interface Props {
  urgency: string;
  recommendation: string;
}

const URGENCY_CONFIG: Record<string, {
  gradient: string;
  iconBg: string;
  textColor: string;
  subtitleColor: string;
  borderColor: string;
  bg: string;
  icon: string;
  label: string;
}> = {
  routine: {
    gradient: "from-emerald-500 to-green-600",
    iconBg: "bg-emerald-100",
    textColor: "text-emerald-800",
    subtitleColor: "text-emerald-600",
    borderColor: "border-emerald-200",
    bg: "bg-emerald-50/50",
    icon: "M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z",
    label: "Routine Follow-up",
  },
  soon: {
    gradient: "from-amber-500 to-orange-500",
    iconBg: "bg-amber-100",
    textColor: "text-amber-800",
    subtitleColor: "text-amber-600",
    borderColor: "border-amber-200",
    bg: "bg-amber-50/50",
    icon: "M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
    label: "Schedule Soon",
  },
  urgent: {
    gradient: "from-red-500 to-red-700",
    iconBg: "bg-red-100",
    textColor: "text-red-800",
    subtitleColor: "text-red-600",
    borderColor: "border-red-200",
    bg: "bg-red-50/50",
    icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z",
    label: "Urgent Referral",
  },
  emergency: {
    gradient: "from-red-600 to-rose-800",
    iconBg: "bg-red-200",
    textColor: "text-red-900",
    subtitleColor: "text-red-700",
    borderColor: "border-red-300",
    bg: "bg-red-100/60",
    icon: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z",
    label: "Emergency Referral",
  },
};

export default function ReferralBanner({ urgency, recommendation }: Props) {
  const key = urgency.toLowerCase();
  const config = URGENCY_CONFIG[key] ?? URGENCY_CONFIG["routine"];

  return (
    <div className={`card-elevated overflow-hidden fade-in-up stagger-2 ${config.bg} border ${config.borderColor}`}>
      <div className="flex items-start gap-4 p-6">
        {/* Icon */}
        <div className={`flex-shrink-0 w-11 h-11 rounded-xl ${config.iconBg} flex items-center justify-center`}>
          <svg className={`w-6 h-6 ${config.subtitleColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d={config.icon} />
          </svg>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-xs font-bold uppercase tracking-wider ${config.subtitleColor}`}>
              Referral
            </span>
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gradient-to-r ${config.gradient} text-white`}>
              {config.label}
            </span>
          </div>
          <p className={`text-sm leading-relaxed ${config.textColor} mt-2`}>
            {recommendation}
          </p>
        </div>
      </div>
    </div>
  );
}
