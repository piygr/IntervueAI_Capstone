// page.tsx
"use client";

import { useSearchParams } from 'next/navigation';
import { CloseIcon } from "@/components/CloseIcon";
import { NoAgentNotification } from "@/components/NoAgentNotification";
import TranscriptionView, { Segment } from "@/components/TranscriptionView";
import useCombinedTranscriptions from '@/hooks/useCombinedTranscriptions';
import {
  BarVisualizer,
  DisconnectButton,
  RoomAudioRenderer,
  RoomContext,
  VideoTrack,
  VoiceAssistantControlBar,
  useVoiceAssistant,
} from "@livekit/components-react";
import { AnimatePresence, motion } from "framer-motion";
import { Room, RoomEvent, AudioCaptureOptions } from "livekit-client";
import { useCallback, useEffect, useState } from "react";
import InterviewWithEditor from "@/components/InterviewWithEditor";

export default function InterviewPage({ params }: { params: { interviewId: string } }) {
  const searchParams = useSearchParams();
  const serverUrl = searchParams.get('s') || '';
  const roomToken = searchParams.get('r') || '';
  const [room] = useState(new Room());

  return (
    <main data-lk-theme="default" className="h-full grid content-center bg-[var(--lk-bg)]">
      <RoomContext.Provider value={room}>
        <InterviewContent room={room} serverUrl={serverUrl} roomToken={roomToken} />
      </RoomContext.Provider>
    </main>
  );
}

function InterviewContent({ room, serverUrl, roomToken }: { room: Room, serverUrl: string, roomToken: string }) {
  const strans = useCombinedTranscriptions();
  const [transcripts, setTranscripts] = useState<Segment[]>([]);

  useEffect(() => {
    setTranscripts(strans);
  }, [strans]);

  const onConnectButtonClicked = useCallback(async () => {
    const audioCaptureOptions: AudioCaptureOptions = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      voiceIsolation: true
    };
    await room.connect(serverUrl, roomToken);
    await room.localParticipant.setMicrophoneEnabled(true, audioCaptureOptions);
  }, [room, serverUrl, roomToken]);

  useEffect(() => {
    onConnectButtonClicked();
    room.on(RoomEvent.MediaDevicesError, onDeviceFailure);
    return () => {
      room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
    };
  }, [onConnectButtonClicked, room]);

  return (
    <div className="flex w-screen h-screen">
      <InterviewWithEditor
        initialCode={"// Write your code here"}
        onSubmit={code => alert("Submitted code:\n" + code)}
        codeLanguage="js"
        highlight={true}
      >
        <SimpleVoiceAssistant room={room} serverUrl={serverUrl} participantToken={roomToken} transcripts={transcripts} />
      </InterviewWithEditor>
    </div>
  );
}

function SimpleVoiceAssistant({ room, serverUrl, participantToken, transcripts, editorOpen }: {
  room: Room,
  serverUrl: string,
  participantToken: string,
  transcripts: Segment[],
  editorOpen?: boolean;
}) {
  const { state: agentState } = useVoiceAssistant();

  const onConnectButtonClicked = useCallback(async () => {
    const audioCaptureOptions: AudioCaptureOptions = {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      voiceIsolation: true
    };
    await room.connect(serverUrl, participantToken);
    await room.localParticipant.setMicrophoneEnabled(true, audioCaptureOptions);
  }, [room, serverUrl, participantToken]);

  useEffect(() => {
    room.on(RoomEvent.MediaDevicesError, onDeviceFailure);
    return () => {
      room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
    };
  }, [room]);

  return (
    <AnimatePresence mode="wait">
      {agentState === "disconnected" ? (
        <motion.div key="disconnected" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} transition={{ duration: 0.3 }} className="grid items-center justify-center h-full gap-4" />
      ) : (
        <motion.div key="connected" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }} className="flex flex-col items-center gap-4 h-full">
          <AgentVisualizer />
          <div className="flex-1 min-w-0">
            <TranscriptionView transcripts={transcripts} fullWidth={editorOpen} />
          </div>
          <div className="w-full">
            <ControlBar onConnectButtonClicked={onConnectButtonClicked} />
          </div>
          <RoomAudioRenderer />
          <NoAgentNotification state={agentState} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function AgentVisualizer() {
  const { state: agentState, videoTrack, audioTrack } = useVoiceAssistant();
  if (videoTrack) {
    return (
      <div className="h-[512px] w-[512px] rounded-lg overflow-hidden">
        <VideoTrack trackRef={videoTrack} />
      </div>
    );
  }
  return (
    <div className="h-[300px] w-full">
      <BarVisualizer state={agentState} barCount={5} trackRef={audioTrack} className="agent-visualizer" options={{ minHeight: 24 }} />
    </div>
  );
}

function ControlBar(props: { onConnectButtonClicked: () => void }) {
  const { state: agentState } = useVoiceAssistant();
  return (
    <div className="relative h-[60px]">
      <AnimatePresence>
        {agentState === "disconnected" && (
          <motion.button initial={{ opacity: 0, top: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, top: "-10px" }} transition={{ duration: 1 }} className="uppercase absolute left-1/2 -translate-x-1/2 px-4 py-2 bg-white text-black rounded-md" onClick={() => props.onConnectButtonClicked()}>
            Start a conversation
          </motion.button>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {agentState !== "disconnected" && agentState !== "connecting" && (
          <motion.div initial={{ opacity: 0, top: "10px" }} animate={{ opacity: 1, top: 0 }} exit={{ opacity: 0, top: "-10px" }} transition={{ duration: 0.4 }} className="flex h-8 absolute left-1/2 -translate-x-1/2 justify-center">
            <VoiceAssistantControlBar controls={{ leave: false }} />
            <DisconnectButton>
              <CloseIcon />
            </DisconnectButton>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function onDeviceFailure(error: Error) {
  console.error(error);
  alert("Error acquiring camera or microphone permissions. Please grant permissions and reload.");
}
