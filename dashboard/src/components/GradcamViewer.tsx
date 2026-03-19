interface Props {
  originalSrc: string;
  gradcamBase64: string;
}

export default function GradcamViewer({ originalSrc, gradcamBase64 }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div>
        <h4 className="text-sm font-medium text-gray-500 mb-2">
          Original Image
        </h4>
        <img
          src={originalSrc}
          alt="Original fundus"
          className="w-full rounded-lg shadow border"
        />
      </div>
      <div>
        <h4 className="text-sm font-medium text-gray-500 mb-2">
          GradCAM Heatmap
        </h4>
        <img
          src={`data:image/png;base64,${gradcamBase64}`}
          alt="GradCAM heatmap"
          className="w-full rounded-lg shadow border"
        />
      </div>
    </div>
  );
}
