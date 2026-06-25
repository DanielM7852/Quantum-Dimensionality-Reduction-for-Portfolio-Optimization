"use client";

type Props = {
  correlationThreshold: number;
  iterations: number;
  rotationScale: number;
  onCorrelationThresholdChange: (value: number) => void;
  onIterationsChange: (value: number) => void;
  onRotationScaleChange: (value: number) => void;
  disabled?: boolean;
};

const ROTATION_PRESETS = [
  { label: "pi/72 (fine)", value: Math.PI / 72 },
  { label: "pi/36 (default)", value: Math.PI / 36 },
  { label: "pi/18 (strong)", value: Math.PI / 18 },
] as const;

function formatRotation(value: number): string {
  const match = ROTATION_PRESETS.find((p) => Math.abs(p.value - value) < 1e-6);
  if (match) return match.label;
  return value.toFixed(4);
}

export function EntangleDRSliders({
  correlationThreshold,
  iterations,
  rotationScale,
  onCorrelationThresholdChange,
  onIterationsChange,
  onRotationScaleChange,
  disabled = false,
}: Props) {
  const rotationIndex = ROTATION_PRESETS.findIndex(
    (p) => Math.abs(p.value - rotationScale) < 1e-6
  );
  const safeRotationIndex = rotationIndex >= 0 ? rotationIndex : 1;

  return (
    <div className="rounded-xl border border-slate-300 bg-white p-4 text-black">
      <h3 className="mb-1 text-sm font-semibold text-black">EntangleDR hyperparameters</h3>
      <p className="mb-4 text-xs text-black">
        These sliders control only the EntangleDR method. PCA, Kernel-PCA, and No-DR baselines
        are unchanged. When you click Run analysis, the backend rebuilds EntangleDR with your
        settings, runs purged cross-validation, and returns updated Sharpe, accuracy, and equity
        curves for all 12 model combinations.
      </p>

      <div className="space-y-6">
        <label className="block">
          <div className="mb-1 flex justify-between text-sm text-black">
            <span className="font-medium">tau: correlation threshold</span>
            <span className="font-mono">{correlationThreshold.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={0.9}
            step={0.05}
            value={correlationThreshold}
            disabled={disabled}
            onChange={(e) => onCorrelationThresholdChange(Number(e.target.value))}
            className="w-full accent-indigo-600"
          />
          <div className="mt-1 flex justify-between text-xs text-black">
            <span>0.50 (more pairs)</span>
            <span>0.90 (fewer pairs)</span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-black">
            EntangleDR measures absolute correlation between each feature pair. Only pairs above
            tau receive a Givens rotation. Lower tau rotates more pairs (broader mixing). Higher
            tau focuses on strongly correlated features only, such as moving-average ratios.
          </p>
        </label>

        <label className="block">
          <div className="mb-1 flex justify-between text-sm text-black">
            <span className="font-medium">Iterations (I)</span>
            <span className="font-mono">{iterations}</span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={iterations}
            disabled={disabled}
            onChange={(e) => onIterationsChange(Number(e.target.value))}
            className="w-full accent-indigo-600"
          />
          <div className="mt-1 flex justify-between text-xs text-black">
            <span>1 pass</span>
            <span>10 passes</span>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-black">
            Each iteration scans all feature pairs again and applies rotations where correlation
            exceeds tau. More iterations compound the transformation, which can further separate
            shared variance before the top components are selected.
          </p>
        </label>

        <label className="block">
          <div className="mb-1 flex justify-between text-sm text-black">
            <span className="font-medium">Rotation scale (R)</span>
            <span className="font-mono">{formatRotation(rotationScale)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={ROTATION_PRESETS.length - 1}
            step={1}
            value={safeRotationIndex}
            disabled={disabled}
            onChange={(e) => {
              const idx = Number(e.target.value);
              onRotationScaleChange(ROTATION_PRESETS[idx].value);
            }}
            className="w-full accent-indigo-600"
          />
          <div className="mt-1 flex justify-between text-xs text-black">
            {ROTATION_PRESETS.map((p) => (
              <span key={p.label}>{p.label.split(" ")[0]}</span>
            ))}
          </div>
          <p className="mt-2 text-xs leading-relaxed text-black">
            For each qualifying pair, rotation angle equals R times the correlation coefficient.
            A larger R applies a stronger twist in that feature plane. Smaller R makes gentler
            adjustments. After rotation, EntangleDR keeps the k components with highest variance.
          </p>
        </label>
      </div>
    </div>
  );
}
