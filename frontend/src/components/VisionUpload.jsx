import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, X, ArrowRight, Loader2, Sparkles } from 'lucide-react';

export default function VisionUpload({ onAnalyze, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [userQuestion, setUserQuestion] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPEG, PNG, WEBP).');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10 MB limit.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleRemove = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    onAnalyze(selectedFile, userQuestion, previewUrl);
  };

  return (
    <div className="max-w-2xl mx-auto w-full bg-white rounded-2xl p-6 sm:p-8 border border-[#EEDFD7] shadow-sm text-center animate-fade-in space-y-6">
      <div className="space-y-1.5">
        <h2 className="text-xl sm:text-2xl font-semibold text-[#29242A]">
          Show a Product
        </h2>
        <p className="text-xs sm:text-sm text-[#756873]">
          Upload an image to identify the product and retrieve relevant BIS standards.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Upload Dropzone */}
        {!selectedFile ? (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 cursor-pointer transition-all duration-200 flex flex-col items-center justify-center space-y-3 ${
              dragActive
                ? 'border-[#D98268] bg-[#FBE3D5]'
                : 'border-[#EEDFD7] hover:border-[#D98268]/60 bg-[#FBE3D5]/40 hover:bg-[#FBE3D5]/70'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => handleFileChange(e.target.files[0])}
              className="hidden"
            />
            <div className="p-3.5 rounded-full bg-white border border-[#EEDFD7] text-[#4B1248] shadow-sm">
              <UploadCloud className="w-7 h-7 text-[#4B1248]" />
            </div>
            <div>
              <p className="text-sm font-semibold text-[#29242A]">
                Upload product image, or <span className="text-[#4B1248] underline">browse</span>
              </p>
              <p className="text-xs text-[#756873] mt-1">
                Supports JPEG, PNG, WEBP (Max 10 MB)
              </p>
            </div>
          </div>
        ) : (
          /* Image Preview Box */
          <div className="relative rounded-2xl border border-[#EEDFD7] p-4 flex flex-col sm:flex-row items-center gap-4 bg-[#FFF9F5] text-left">
            <div className="w-32 h-32 rounded-xl overflow-hidden bg-white border border-[#EEDFD7] shrink-0 relative flex items-center justify-center shadow-xs">
              <img
                src={previewUrl}
                alt="Product preview"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-center space-x-2 text-xs font-semibold text-[#29242A]">
                <ImageIcon className="w-4 h-4 text-[#4B1248]" />
                <span className="truncate">{selectedFile.name}</span>
              </div>
              <p className="text-xs text-[#756873] font-mono">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis
              </p>

              <button
                type="button"
                onClick={handleRemove}
                className="inline-flex items-center space-x-1 text-xs text-rose-600 hover:text-rose-700 mt-2 font-medium transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
                <span>Remove or replace image</span>
              </button>
            </div>
          </div>
        )}

        {/* Optional Question Input */}
        <div className="text-left space-y-1.5">
          <label className="text-xs font-medium text-[#756873]">
            Optional Question (e.g. "Is BIS certification required for this product?")
          </label>
          <input
            type="text"
            value={userQuestion}
            onChange={(e) => setUserQuestion(e.target.value)}
            placeholder="Ask something specific about this product..."
            disabled={loading}
            className="w-full bg-[#FFF9F5] rounded-xl px-4 py-2.5 text-xs sm:text-sm text-[#29242A] placeholder-[#756873]/60 outline-none border border-[#EEDFD7] focus:border-[#4B1248] transition-colors"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!selectedFile || loading}
          className={`w-full py-3 rounded-xl font-medium text-sm transition-all duration-200 flex items-center justify-center space-x-2 cursor-pointer ${
            selectedFile && !loading
              ? 'bg-[#4B1248] hover:bg-[#64175F] text-white shadow-xs'
              : 'bg-[#EEDFD7]/50 text-[#756873] cursor-not-allowed border border-[#EEDFD7]'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin text-[#2864E8]" />
              <span>Analyzing product & checking BIS evidence...</span>
            </>
          ) : (
            <>
              <span>Analyze Image</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </form>
    </div>
  );
}
