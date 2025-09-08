import React, { useState, useRef } from 'react';
import { 
  Card, 
  Button, 
  Text, 
  Badge, 
  Progress, 
  Box, 
  VStack, 
  HStack, 
  Image, 
  useToast,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure
} from '@chakra-ui/react';
import { Camera, Upload, CheckCircle, AlertTriangle, Info } from 'lucide-react';

const AIDrugVerification = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef();
  const cameraRef = useRef();
  const toast = useToast();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setImagePreview(e.target.result);
        setVerificationResult(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCameraCapture = () => {
    // Simulate camera capture
    onOpen();
  };

  const simulateVerification = async () => {
    if (!selectedImage) {
      toast({
        title: "No Image Selected",
        description: "Please upload or capture an image first",
        status: "warning",
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    setIsVerifying(true);
    
    // Simulate API call delay
    setTimeout(() => {
      const mockResult = {
        verified: Math.random() > 0.2, // 80% success rate
        confidence_score: (0.7 + Math.random() * 0.25).toFixed(2),
        detected_drug_name: "Amoxicillin 500mg",
        detected_dosage: "500mg",
        label_quality: ["Excellent", "Good", "Fair"][Math.floor(Math.random() * 3)],
        color_match: ["Perfect", "Good", "Acceptable"][Math.floor(Math.random() * 3)],
        shape_match: ["Exact", "Close", "Similar"][Math.floor(Math.random() * 3)],
        recommendations: [
          "Label is clearly visible and matches expected medication",
          "Dosage information is accurate",
          "Expiry date is within acceptable range"
        ]
      };
      
      setVerificationResult(mockResult);
      setIsVerifying(false);
      
      toast({
        title: "Verification Complete",
        description: `Drug verified with ${(mockResult.confidence_score * 100).toFixed(0)}% confidence`,
        status: mockResult.verified ? "success" : "error",
        duration: 5000,
        isClosable: true,
      });
    }, 2000);
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.9) return "green";
    if (score >= 0.7) return "yellow";
    return "red";
  };

  const getQualityColor = (quality) => {
    switch (quality) {
      case "Excellent": return "green";
      case "Good": return "blue";
      case "Fair": return "orange";
      default: return "gray";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center py-6">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          AI Drug Verification
        </h2>
        <p className="text-lg text-gray-600 mt-2">
          Verify medications using advanced computer vision and AI
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Panel - Image Upload & Camera */}
        <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
          <div className="space-y-4">
            {/* Image Preview */}
            {imagePreview && (
              <div className="relative">
                <img 
                  src={imagePreview} 
                  alt="Drug preview" 
                  className="max-h-72 w-full object-contain rounded-lg border border-gray-200"
                />
                <button
                  onClick={() => {
                    setSelectedImage(null);
                    setImagePreview(null);
                    setVerificationResult(null);
                  }}
                  className="absolute top-2 right-2 bg-red-500 text-white px-2 py-1 rounded text-sm hover:bg-red-600"
                >
                  Remove
                </button>
              </div>
            )}

            {/* Upload Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center justify-center gap-2 px-4 py-3 border-2 border-blue-300 text-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
              >
                📤 Upload Image
              </button>
              <button
                onClick={handleCameraCapture}
                className="flex items-center justify-center gap-2 px-4 py-3 border-2 border-green-300 text-green-600 rounded-lg hover:bg-green-50 transition-colors"
              >
                📷 Take Photo
              </button>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: 'none' }}
            />

            {/* Verify Button */}
            <button
              onClick={simulateVerification}
              disabled={!selectedImage || isVerifying}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 transition-all disabled:opacity-50"
            >
              {isVerifying ? "Verifying..." : "🤖 Verify Drug"}
            </button>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="bg-white/70 backdrop-blur-sm border border-gray-200 shadow-lg rounded-xl p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Verification Results</h3>

          {verificationResult ? (
            <div className="space-y-4">
              {/* Verification Status */}
              <div className="flex justify-between items-center">
                <span className="font-semibold">Status:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  verificationResult.verified ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {verificationResult.verified ? "✅ VERIFIED" : "❌ NOT VERIFIED"}
                </span>
              </div>

              {/* Confidence Score */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold">Confidence:</span>
                  <span className="font-bold text-lg">
                    {(verificationResult.confidence_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full ${
                      verificationResult.confidence_score >= 0.9 ? 'bg-green-500' :
                      verificationResult.confidence_score >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{width: `${verificationResult.confidence_score * 100}%`}}
                  />
                </div>
              </div>

              {/* Drug Details */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="font-semibold">Drug Name:</span>
                  <span>{verificationResult.detected_drug_name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-semibold">Dosage:</span>
                  <span>{verificationResult.detected_dosage}</span>
                </div>
              </div>

              {/* Quality Metrics */}
              <div className="space-y-2">
                <p className="font-semibold">Quality Assessment:</p>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Label Quality:</span>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      verificationResult.label_quality === 'Excellent' ? 'bg-green-100 text-green-800' :
                      verificationResult.label_quality === 'Good' ? 'bg-blue-100 text-blue-800' :
                      'bg-orange-100 text-orange-800'
                    }`}>
                      {verificationResult.label_quality}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Color Match:</span>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      verificationResult.color_match === 'Perfect' ? 'bg-green-100 text-green-800' :
                      verificationResult.color_match === 'Good' ? 'bg-blue-100 text-blue-800' :
                      'bg-orange-100 text-orange-800'
                    }`}>
                      {verificationResult.color_match}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Shape Match:</span>
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      verificationResult.shape_match === 'Exact' ? 'bg-green-100 text-green-800' :
                      verificationResult.shape_match === 'Close' ? 'bg-blue-100 text-blue-800' :
                      'bg-orange-100 text-orange-800'
                    }`}>
                      {verificationResult.shape_match}
                    </span>
                  </div>
                </div>
              </div>

              {/* Recommendations */}
              <div>
                <p className="font-semibold mb-2">Recommendations:</p>
                <div className="space-y-2">
                  {verificationResult.recommendations.map((rec, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <span className="text-green-500 mt-1">✓</span>
                      <span className="text-sm">{rec}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🤖</div>
              <p className="text-gray-500 text-lg">Upload or capture an image to verify</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIDrugVerification;
