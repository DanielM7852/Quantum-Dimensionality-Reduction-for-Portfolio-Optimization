"use client";

import { useEffect, useState } from "react";

const DISCLAIMER =
  "This tool is for educational and research purposes only. It is not investment advice. Past simulated performance does not guarantee future results. Data from Yahoo Finance may be delayed or inaccurate. Do not use this to make real trading decisions.";

export function DisclaimerBanner() {
  const [showModal, setShowModal] = useState(false);

  useEffect(() => {
    const seen = localStorage.getItem("disclaimer_seen_v1");
    if (!seen) setShowModal(true);
  }, []);

  const dismiss = () => {
    localStorage.setItem("disclaimer_seen_v1", "1");
    setShowModal(false);
  };

  return (
    <>
      <div className="sticky top-0 z-40 border-b border-amber-300 bg-amber-50 px-4 py-2 text-center text-sm text-amber-950">
        <strong>Research demo only.</strong> {DISCLAIMER}
      </div>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="max-w-lg rounded-xl bg-white p-6 shadow-xl">
            <h2 className="mb-3 text-lg font-bold text-slate-900">Important Disclaimer</h2>
            <p className="mb-6 text-sm leading-relaxed text-slate-600">{DISCLAIMER}</p>
            <button
              onClick={dismiss}
              className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
            >
              I understand. Continue to demo
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export function DisclaimerFooter() {
  return (
    <footer className="mt-auto border-t border-slate-200 bg-slate-50 px-4 py-6 text-center text-xs text-slate-500">
      EntangleDR Lab · Not financial advice · For educational use only
    </footer>
  );
}
