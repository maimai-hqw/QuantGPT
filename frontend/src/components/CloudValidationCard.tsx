import type { CloudValidationResult, CloudCheck } from "../api/cloud";
import { useColorMode } from "../contexts/ColorModeContext";

interface Props {
  result: CloudValidationResult;
  cloudUrl: string;
}

function num(n: number | null): string {
  return n != null ? n.toFixed(4) : "—";
}

function pct(n: number | null): string {
  return n != null ? (n * 100).toFixed(2) + "%" : "—";
}

export default function CloudValidationCard({ result, cloudUrl }: Props) {
  const { isDark } = useColorMode();
  const checks = result.checks ?? [];
  const passCount = checks.filter((c: CloudCheck) => c.result === "PASS").length;
  const totalCount = checks.length;
  const isActive = result.status === "active";
  const m = result.is;

  return (
    <div className={`rounded-xl border ${isDark ? "border-gray-700" : "border-gray-200"} ${isDark ? "bg-gray-900" : "bg-white"} p-4`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium" style={{ color: isDark ? "#e5e7eb" : "#374151" }}>
            QuantGPT Cloud 独立验证
          </span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
            isActive
              ? isDark ? "bg-emerald-500/20 text-emerald-400" : "bg-emerald-50 text-emerald-700"
              : isDark ? "bg-red-500/20 text-red-400" : "bg-red-50 text-red-700"
          }`}>
            {isActive ? "通过" : "未通过"}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <a
            href={cloudUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={`text-xs ${isDark ? "text-blue-400 hover:text-blue-300" : "text-blue-600 hover:text-blue-500"}`}
          >
            quant-gpt.com
          </a>
          <span className={`text-xs font-medium ${
            isActive
              ? isDark ? "text-emerald-400" : "text-emerald-600"
              : isDark ? "text-amber-400" : "text-amber-600"
          }`}>
            {passCount}/{totalCount} 项检查通过
          </span>
        </div>
      </div>

      <div className="grid grid-cols-5 gap-3 mb-3">
        <Metric label="IC Mean" value={num(m.ic_mean)} highlight={m.ic_mean != null && Math.abs(m.ic_mean) > 0.015} isDark={isDark} />
        <Metric label="IC IR" value={num(m.ic_ir)} highlight={m.ic_ir != null && Math.abs(m.ic_ir) > 0.15} isDark={isDark} />
        <Metric label="Turnover" value={pct(m.turnover)} highlight={false} isDark={isDark} />
        <Metric label="Sharpe" value={num(m.sharpe)} highlight={m.sharpe != null && m.sharpe >= 1} isDark={isDark} />
        <Metric label="Fitness" value={num(m.fitness)} highlight={m.fitness != null && m.fitness >= 1} isDark={isDark} />
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <Metric label="覆盖率" value={pct(m.coverage)} highlight={false} isDark={isDark} />
        <Metric label="数据天数" value={m.data_days != null ? String(m.data_days) : "—"} highlight={m.data_days != null && m.data_days >= 120} isDark={isDark} />
        <Metric label="最大相关性" value={num(m.max_correlation)} highlight={false} isDark={isDark} />
      </div>

      {result.reject_reason && (
        <div className={`text-xs px-3 py-2 rounded-lg mb-3 ${isDark ? "bg-red-500/10 text-red-400" : "bg-red-50 text-red-600"}`}>
          未通过原因：{result.reject_reason}
        </div>
      )}

      <div className="space-y-1.5">
        {checks.map((check: CloudCheck) => (
          <div key={check.name} className={`flex items-center justify-between px-3 py-1.5 rounded-lg text-xs ${isDark ? "bg-gray-800/50" : "bg-gray-50"}`}>
            <div className="flex items-center gap-2">
              <span className={check.result === "PASS"
                ? isDark ? "text-emerald-400" : "text-emerald-500"
                : isDark ? "text-red-400" : "text-red-500"
              }>
                {check.result === "PASS" ? "✓" : "✗"}
              </span>
              <span className={isDark ? "text-gray-300" : "text-gray-700"}>{check.name}</span>
            </div>
            <div className="flex items-center gap-3">
              <span className={`font-mono font-medium ${
                check.result === "PASS"
                  ? isDark ? "text-emerald-400" : "text-emerald-600"
                  : isDark ? "text-red-400" : "text-red-500"
              }`}>
                {check.value != null ? check.value.toFixed(4) : "—"}
              </span>
              <span className={isDark ? "text-gray-500" : "text-gray-400"}>
                {check.limit}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value, highlight, isDark }: { label: string; value: string; highlight: boolean; isDark: boolean }) {
  return (
    <div className={`rounded-lg p-2.5 ${isDark ? "bg-gray-800" : "bg-gray-50"}`}>
      <p className={`text-xs ${isDark ? "text-gray-400" : "text-gray-500"}`}>{label}</p>
      <p className={`text-lg font-semibold ${
        highlight
          ? isDark ? "text-emerald-400" : "text-emerald-600"
          : isDark ? "text-gray-100" : "text-gray-900"
      }`}>
        {value}
      </p>
    </div>
  );
}
