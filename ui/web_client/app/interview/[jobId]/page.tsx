// page.tsx
"use client";

import { useSearchParams, useRouter } from 'next/navigation';
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
import { Room, RoomEvent, AudioCaptureOptions, Participant, DataPacket_Kind, RemoteParticipant } from "livekit-client";
import { useMemo, useRef, useEffect, useCallback, useState  } from "react";
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import InterviewWithEditor from "@/components/InterviewWithEditor";
import { useRoomContext } from "@livekit/components-react";
import debounce from 'lodash/debounce';

const encoder = new TextEncoder()
const decoder = new TextDecoder()

export default function InterviewPage({ params }: { params: { jobId: string } }) {
  const searchParams = useSearchParams();
  const serverUrl = searchParams.get('s') || '';
  const roomToken = searchParams.get('r') || '';
  const interviewId = searchParams.get('i') || '';
  const [room] = useState(new Room());

  return (
    <main data-lk-theme="default" className="h-full grid content-center bg-[var(--lk-bg)]">
      <RoomContext.Provider value={room}>
        <InterviewContent room={room} serverUrl={serverUrl} roomToken={roomToken} jobId={params.jobId} interviewId={interviewId} />
      </RoomContext.Provider>
    </main>
  );
}

function InterviewContent({ room, serverUrl, roomToken, jobId, interviewId }: { room: Room, serverUrl: string, roomToken: string, jobId: string, interviewId: string }) {
  const strans = useCombinedTranscriptions();
  const [transcripts, setTranscripts] = useState<Segment[]>([]);
  const [code, setCode] = useState("// Initial code from UI");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setTranscripts(strans);
  }, [strans]);

  useEffect(() => {
    const handleData = (payload: Uint8Array, participant?: Participant, kind?: DataPacket_Kind, topic?: string) => {
      const text = decoder.decode(payload);
      if (topic === "interview-status") {
        if (text === "interview-ended") {
          console.log('feedback jobId: ' + jobId)
          console.log('feedback participant: ' + participant?.identity)
          console.log('feedback interviewId: ' + interviewId)
          router.push(`/interview/${jobId}/feedback?i=${interviewId}`);
        }
      } else if (topic === "code-editor") {
        const updatedCode = text + "\n\n// Type your code below this:\n";
        setCode(updatedCode);
      }
    };
  
    room.on(RoomEvent.DataReceived, handleData);
    return () => {
      room.off(RoomEvent.DataReceived, handleData);
    };
  }, [room, router, jobId, interviewId]);

  const debouncedHandleStreamingCode = useMemo(() => {
    const debounced = debounce((codeText: string) => {
      const data = encoder.encode(codeText);
      room.localParticipant.publishData(
        data,
        { reliable: true, topic: "code-editor" }
      );
    }, 1000);
    return debounced;
  }, [room]);

  const handleSubmit = useCallback(
    (codeText: string) => {
      setIsSubmitting(true);
      try {
        debouncedHandleStreamingCode.flush();
        codeText = codeText + "<--final code-->";
        const data = encoder.encode(codeText);
        room.localParticipant.publishData(
          data,
          { reliable: true, topic: "code-editor" }
        );
        toast.success("Code submitted successfully!");
      } catch (error) {
        console.error("Error during submission:", error);
        toast.error("Failed to submit code.");
      } finally {
        setIsSubmitting(false);
      }
    },
    [room, debouncedHandleStreamingCode]
  );

  const handleCodeChange = useCallback((newCode: string) => {
    setCode(newCode);
    debouncedHandleStreamingCode(newCode);
  }, [debouncedHandleStreamingCode]);

  useEffect(() => {
    return () => {
      debouncedHandleStreamingCode.cancel();
    };
  }, [debouncedHandleStreamingCode]);

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
        code={code}
        onCodeChange={handleCodeChange}
        onSubmit={handleSubmit}
        codeLanguage="js"
        highlight={true}
        isSubmitting={isSubmitting}
      >
        <SimpleVoiceAssistant room={room} serverUrl={serverUrl} participantToken={roomToken} transcripts={transcripts} />
      </InterviewWithEditor>
      <ToastContainer position="top-center" autoClose={2000} aria-label="Notification" />
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