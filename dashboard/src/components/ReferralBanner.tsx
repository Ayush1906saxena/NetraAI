interface Props {
  urgency: string;
  recommendation: string;
}

const URGENCY_STYLES: Record<string, string> = {
  routine: "bg-green-50 border-[#28a745] text-green-800",
  soon: "bg-amber-50 border-[#ffc107] text-amber-800",
  urgent: "bg-red-50 border-[#dc3545] text-red-800",
};

export default function ReferralBanner({ urgency, recommendation }: Props) {
  const style =
    URGENCY_STYLES[urgency.toLowerCase()] ?? URGENCY_STYLES["routine"];

  return (
    <div className={`rounded-xl border-l-4 p-4 ${style}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className="font-semibold text-sm uppercase tracking-wide">
          Referral: {urgency}
        </span>
      </div>
      <p className="text-sm">{recommendation}</p>
    </div>
  );
}
