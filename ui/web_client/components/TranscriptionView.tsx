// TranscriptionView.tsx

import React, { useRef, useEffect } from "react";

// Define Segment type. Adjust fields to match your hook/provider.
export interface Segment {
  id: string;
  role: string //"assistant" | "user";
  text: string;
}

interface TranscriptionViewProps {
  transcripts: Segment[];
  fullWidth?: boolean; 
}

export default function TranscriptionView({ transcripts, fullWidth }: TranscriptionViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever transcripts change
  useEffect(() => {
    containerRef.current?.scrollTo(0, containerRef.current.scrollHeight);
  }, [transcripts]);

  return (
    <div className={
      fullWidth
        ? "relative h-[200px] w-full min-w-0"
        : "relative h-[200px] w-[512px] max-w-[90vw] mx-auto"
    }>
      {/* Fade-out top gradient */}
      <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-[var(--lk-bg)] to-transparent z-10 pointer-events-none" />
      {/* Fade-out bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[var(--lk-bg)] to-transparent z-10 pointer-events-none" />

      {/* Scrollable content */}
      <div
        ref={containerRef}
        className="h-full flex flex-col gap-2 overflow-y-auto px-4 py-8"
      >
        {transcripts.map((segment) => (
          <div
            id={segment.id}
            key={segment.id}
            className={
              segment.role === "assistant"
                ? "p-2 self-start"
                : "bg-gray-800 rounded-md p-2 self-end"
            }
          >
            {segment.text}
          </div>
        ))}
      </div>
    </div>
  );
}
