"use client";

import type { AnalyzeResponse } from "@/lib/api";
import { combosToCsv } from "@/lib/api";

type Props = {
  result: AnalyzeResponse;
  chartId?: string;
};

export function ExportPanel({ result, chartId = "equity-chart" }: Props) {
  const downloadCsv = () => {
    const csv = combosToCsv(result.combos);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `entangledr_${result.meta.ticker}_${result.meta.start_date}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadChartPng = async () => {
    const container = document.getElementById(chartId);
  const svg = container?.querySelector("svg");
    if (!svg) return;

    const svgData = new XMLSerializer().serializeToString(svg);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);

    img.onload = () => {
      canvas.width = img.width || 800;
      canvas.height = img.height || 300;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const pngUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = pngUrl;
        a.download = `entangledr_equity_${result.meta.ticker}.png`;
        a.click();
        URL.revokeObjectURL(pngUrl);
      });
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  return (
    <div className="flex flex-wrap gap-3">
      <button
        type="button"
        onClick={downloadCsv}
        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-black hover:bg-slate-50"
      >
        Export results CSV
      </button>
      <button
        type="button"
        onClick={downloadChartPng}
        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-black hover:bg-slate-50"
      >
        Export equity chart PNG
      </button>
    </div>
  );
}
