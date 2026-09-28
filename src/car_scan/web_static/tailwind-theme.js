window.tailwind = window.tailwind || {};
window.tailwind.config = {
  corePlugins: { preflight: false },
  theme: {
    fontFamily: {
      sans: ["Noto Sans Lao", "Noto Sans Thai", "Inter", "sans-serif"],
    },
    extend: {
      colors: {
        ink: "#0F172A",
        rail: "#0F172A",
        desk: "#F8FAFC",
        panel: "#FFFFFF",
        line: "#E2E8F0",
        muted: "#64748B",
        enamel: { DEFAULT: "#2563EB", ink: "#FFFFFF", bright: "#60A5FA" },
        signal: "#DC2626",
        forest: "#22C55E",
        steel: "#1E293B",
        well: "#0F172A",
        roi: "#2563EB",
      },
      borderRadius: {
        gate: "12px",
      },
      boxShadow: {
        card: "0 4px 16px rgba(15, 23, 42, 0.06)",
      },
    },
  },
};
