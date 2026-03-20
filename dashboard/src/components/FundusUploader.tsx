import { useCallback, useState, useRef } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function FundusUploader({ onFileSelected, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-5">
      {/* Upload zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`group relative rounded-2xl p-10 sm:p-14 text-center cursor-pointer transition-all duration-500 ease-out border-2 border-dashed ${
          dragOver
            ? "border-indigo-400 bg-indigo-50/60 scale-[1.015] shadow-[0_0_40px_rgba(99,102,241,0.15)]"
            : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30 hover:shadow-[0_0_30px_rgba(99,102,241,0.08)]"
        } ${disabled ? "opacity-50 cursor-not-allowed pointer-events-none" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          className="hidden"
          disabled={disabled}
        />

        {/* Animated eye icon */}
        <div className="relative mx-auto w-20 h-20 mb-6">
          {/* Pulse rings */}
          <div className={`absolute inset-0 rounded-full gradient-brand opacity-20 transition-all duration-700 ${
            dragOver ? "scale-[1.6] opacity-10" : "group-hover:scale-[1.3] group-hover:opacity-10"
          }`} />
          <div className={`absolute inset-0 rounded-full gradient-brand opacity-10 transition-all duration-1000 ${
            dragOver ? "scale-[2] opacity-5" : "group-hover:scale-[1.5] group-hover:opacity-5"
          }`} />

          {/* Icon circle */}
          <div className={`relative w-20 h-20 rounded-full gradient-brand flex items-center justify-center shadow-lg transition-all duration-500 ${
            dragOver ? "shadow-[0_8px_30px_rgba(99,102,241,0.4)] scale-110" : "group-hover:shadow-[0_8px_24px_rgba(99,102,241,0.3)] group-hover:scale-105"
          }`}>
            <svg className="w-9 h-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
        </div>

        <p className="text-xl font-semibold text-gray-800">
          {dragOver ? "Drop your image here" : "Upload Fundus Image"}
        </p>
        <p className="text-sm text-gray-400 mt-2">
          Drag & drop or{" "}
          <span className="text-indigo-500 font-medium">click to browse</span>
        </p>

        {/* File type chips */}
        <div className="flex items-center justify-center gap-2 mt-5">
          {["JPEG", "PNG", "TIFF"].map((type) => (
            <span
              key={type}
              className="px-3 py-1 rounded-full bg-gray-100 text-xs font-medium text-gray-500 border border-gray-150"
            >
              {type}
            </span>
          ))}
          <span className="text-xs text-gray-300 ml-1">up to 20 MB</span>
        </div>
      </div>

      {/* Hint */}
      <p className="text-center text-xs text-gray-400">
        No fundus image available? Use a sample from the{" "}
        <span className="text-indigo-400 font-medium">APTOS 2019 dataset</span>{" "}
        for testing.
      </p>
    </div>
  );
}
