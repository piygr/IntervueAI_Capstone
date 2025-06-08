'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const ResumeUploadPopup = ({ jdId, onClose }: { jdId: string; onClose: () => void }) => {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [roomToken, setRoomToken] = useState<string | null>(null);
  const [serverUrl, setServerUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isRedirecting, setIsRedirecting] = useState(false);

  const handleResumeUpload = async () => {
    if (!file) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jobId', jdId);

    try {
      const res = await fetch(process.env.API_HOST + '/start-interview', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      setRoomToken(data.participantToken);
      setServerUrl(data.serverUrl);
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  // Automatically redirect when tokens are available
  useEffect(() => {
    if (roomToken && serverUrl) {
      setIsRedirecting(true);
      const url = `/interview/${jdId}?r=${roomToken}&s=${serverUrl}`;
      router.push(url);
    }
  }, [roomToken, serverUrl, jdId, router]);

  return (
    <>
      {(isRedirecting) && (
        <div className="fixed inset-0 bg-black bg-opacity-60 z-50 flex items-center justify-center">
          <div className="text-white text-xl font-semibold animate-pulse">Preparing Interview Room...</div>
        </div>
      )}

      <div className="fixed inset-0 bg-black bg-opacity-40 flex justify-center items-center z-40">
        <div className="bg-white p-6 rounded-xl shadow-md w-[380px] space-y-5 text-center">
          <h3 className="text-lg font-bold text-gray-900">Upload Resume to Start Interview</h3>

          <input
            type="file"
            accept=".pdf,.doc,.docx"
            onChange={e => e.target.files && setFile(e.target.files[0])}
            className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
          />

          <div className="space-x-2 pt-2">
            <button
              onClick={handleResumeUpload}
              disabled={!file || isUploading}
              className={`px-4 py-2 rounded-md text-white text-sm font-semibold ${
                file ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-gray-400 cursor-not-allowed'
              } transition-colors`}
            >
              {isUploading ? 'Uploading...' : 'Upload & Start Interview'}
            </button>

            <button
              onClick={onClose}
              className="px-4 py-2 rounded-md border border-gray-300 text-gray-700 hover:bg-gray-100 transition-colors text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default ResumeUploadPopup;
