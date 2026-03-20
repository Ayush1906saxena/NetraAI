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
    <div className="space-y-6">
      {/* Upload Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
          dragOver
            ? "border-[#2E86C1] bg-blue-50/50 scale-[1.01] shadow-lg"
            : "border-gray-300 hover:border-[#2E86C1] hover:bg-blue-50/30 hover:shadow-md"
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          className="hidden"
          disabled={disabled}
        />

        <div className="py-4">
          {/* Eye scan icon */}
          <div className="mx-auto w-20 h-20 rounded-full bg-gradient-to-br from-[#1B4F72] to-[#2E86C1] flex items-center justify-center mb-6 shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <p className="text-xl font-semibold text-gray-700">
            Upload Fundus Image
          </p>
          <p className="text-sm text-gray-400 mt-2">
            Drag & drop or click to browse
          </p>
          <p className="text-xs text-gray-300 mt-1">
            Supports JPEG, PNG up to 20 MB
          </p>
        </div>
      </div>

      {/* Sample images hint */}
      <div className="text-center">
        <p className="text-xs text-gray-400">
          Don't have a fundus image? Use a sample from the APTOS 2019 dataset for testing.
        </p>
      </div>
    </div>
  );
}
