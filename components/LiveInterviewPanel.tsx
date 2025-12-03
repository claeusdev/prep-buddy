
import React, { useEffect, useRef, useState } from 'react';
import { X, Mic, MicOff, Loader2, Radio, Minus, Maximize2, Square } from 'lucide-react';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import type { Question } from '@/types';
import { base64ToUint8Array, arrayBufferToBase64, float32ToInt16, decodeAudioData } from '@/services/audioUtils';

interface LiveInterviewPanelProps {
  isOpen: boolean;
  onClose: () => void;
  question: Question;
  timerDisplay: string;
  mode?: 'coding' | 'system-design';
}

const LiveInterviewPanel: React.FC<LiveInterviewPanelProps> = ({
  isOpen,
  onClose,
  question,
  timerDisplay,
  mode = 'coding'
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isMicOn, setIsMicOn] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  // Refs
  const activeRef = useRef(false);
  const sessionRef = useRef<any>(null);
  const inputContextRef = useRef<AudioContext | null>(null);
  const outputContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  useEffect(() => {
    if (isOpen) {
      startSession();
    } else {
      stopSession();
    }
    return () => stopSession();
  }, [isOpen]);

  const startSession = async () => {
    if (activeRef.current) return;

    try {
      setError(null);
      setIsMinimized(false);
      activeRef.current = true;

      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputContextRef.current = inputCtx;
      outputContextRef.current = outputCtx;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (!activeRef.current) {
        stream.getTracks().forEach(t => t.stop());
        inputCtx.close();
        outputCtx.close();
        return;
      }

      streamRef.current = stream;
      const ai = new GoogleGenAI({ apiKey: import.meta.env.VITE_GEMINI_API_KEY });

      const codingInstruction = `
        You are a professional, slightly strict technical interviewer at a top tech company.
        Problem: "${question.title}".
        Description: ${question.description}
        Solution: ${question.officialSolution}
        Goal: Ask the user to explain their thought process. Be concise.
      `;

      const systemDesignInstruction = `
        You are a Senior System Architect conducting a System Design interview.
        Task: "${question.title}".
        Description: ${question.description}
        Solution: ${question.officialSolution}
        Goal: Drive the conversation on high-level design and tradeoffs. Be professional.
      `;

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: mode === 'system-design' ? systemDesignInstruction : codingInstruction,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: mode === 'system-design' ? 'Fenrir' : 'Kore' } },
          },
        },
        callbacks: {
          onopen: () => {
            if (!activeRef.current) return;
            console.log('Gemini Live Session Connected');
            setIsConnected(true);

            try {
              if (inputCtx.state === 'closed') return;
              const source = inputCtx.createMediaStreamSource(stream);
              const processor = inputCtx.createScriptProcessor(4096, 1, 1);
              processor.onaudioprocess = (e) => {
                if (!activeRef.current || !isMicOn) return;
                const inputData = e.inputBuffer.getChannelData(0);
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
                setVolumeLevel(Math.sqrt(sum / inputData.length));
                const pcmInt16 = float32ToInt16(inputData);
                const base64Data = arrayBufferToBase64(pcmInt16.buffer);
                sessionPromise.then((session) => {
                  if (activeRef.current) {
                    session.sendRealtimeInput({
                      media: {
                        mimeType: 'audio/pcm;rate=16000',
                        data: base64Data
                      }
                    });
                  }
                });
              };
              source.connect(processor);
              processor.connect(inputCtx.destination);
              sourceRef.current = source;
              processorRef.current = processor;
            } catch (setupError) {
              console.error("Audio setup failed", setupError);
            }
          },
          onmessage: async (message: LiveServerMessage) => {
            if (!activeRef.current) return;
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio) {
              setAiSpeaking(true);
              const ctx = outputContextRef.current;
              if (!ctx || ctx.state === 'closed') return;
              try {
                const audioData = base64ToUint8Array(base64Audio);
                const audioBuffer = await decodeAudioData(audioData, ctx, 24000);
                const now = ctx.currentTime;
                const startTime = Math.max(nextStartTimeRef.current, now);
                const source = ctx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(ctx.destination);
                source.start(startTime);
                nextStartTimeRef.current = startTime + audioBuffer.duration;
                audioSourcesRef.current.add(source);
                source.onended = () => {
                  audioSourcesRef.current.delete(source);
                  if (audioSourcesRef.current.size === 0) {
                    setAiSpeaking(false);
                  }
                };
              } catch (decodeErr) {
                console.error("Audio decoding error", decodeErr);
              }
            }
            if (message.serverContent?.interrupted) {
              audioSourcesRef.current.forEach(s => { try { s.stop(); } catch (e) { } });
              audioSourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setAiSpeaking(false);
            }
          },
          onclose: () => {
            if (!activeRef.current) return;
            setIsConnected(false);
          },
          onerror: (err) => {
            if (!activeRef.current) return;
            setError("Network error.");
            setIsConnected(false);
          }
        }
      });
      sessionRef.current = sessionPromise;
    } catch (err: any) {
      if (!activeRef.current) return;
      setError(err.message || "Connection Failed.");
    }
  };

  const stopSession = () => {
    activeRef.current = false;
    if (sessionRef.current) {
      sessionRef.current.then((session: any) => { try { session.close(); } catch (e) { } }).catch(() => { });
      sessionRef.current = null;
    }
    audioSourcesRef.current.forEach(s => { try { s.stop(); } catch (e) { } });
    audioSourcesRef.current.clear();
    if (processorRef.current) { try { processorRef.current.disconnect(); } catch (e) { } processorRef.current = null; }
    if (sourceRef.current) { try { sourceRef.current.disconnect(); } catch (e) { } sourceRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; }
    if (inputContextRef.current) { try { inputContextRef.current.close(); } catch (e) { } inputContextRef.current = null; }
    if (outputContextRef.current) { try { outputContextRef.current.close(); } catch (e) { } outputContextRef.current = null; }
    setIsConnected(false);
    setVolumeLevel(0);
    setAiSpeaking(false);
    nextStartTimeRef.current = 0;
  };

  if (!isOpen) return null;

  // MINIMIZED
  if (isMinimized) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <div className="bg-white border-2 border-black shadow-retro flex items-center p-2 gap-3">
          <div className={`w-3 h-3 rounded-full border border-black ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="font-mono text-xs font-bold text-black uppercase">LIVE_SESSION</span>
          <span className="font-mono text-xs bg-black text-white px-2">{timerDisplay}</span>
          <button onClick={() => setIsMinimized(false)} className="p-1 hover:bg-gray-200 border border-transparent hover:border-black"><Maximize2 size={12} /></button>
        </div>
      </div>
    );
  }

  // MAXIMIZED
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col w-80 bg-[#f0f0f0] border-2 border-black shadow-retro">
      {/* Window Header */}
      <div className="bg-black text-white px-2 py-1 flex justify-between items-center border-b-2 border-black">
        <span className="font-mono text-xs font-bold uppercase">Voice_Link v1.0</span>
        <div className="flex gap-1">
          <button onClick={() => setIsMinimized(true)} className="bg-white text-black w-4 h-4 flex items-center justify-center border border-gray-500 hover:bg-gray-200"><Minus size={10} /></button>
          <button onClick={onClose} className="bg-white text-black w-4 h-4 flex items-center justify-center border border-gray-500 hover:bg-red-500 hover:text-white"><X size={10} /></button>
        </div>
      </div>

      {/* Visualizer Screen */}
      <div className="h-40 bg-black border-b-2 border-black relative flex items-center justify-center overflow-hidden">
        {/* Scanlines */}
        <div className="absolute inset-0 pointer-events-none opacity-10" style={{ backgroundImage: 'linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))', backgroundSize: '100% 2px, 3px 100%' }}></div>

        {!isConnected && !error && (
          <div className="flex flex-col items-center gap-2 text-green-500 font-mono">
            <Loader2 className="animate-spin" size={24} />
            <span className="text-xs uppercase">Est_Connection...</span>
          </div>
        )}

        {error && (
          <div className="text-red-500 font-mono text-xs text-center px-4">
            <span className="block mb-2">CONN_ERR: {error}</span>
            <button onClick={() => { stopSession(); startSession(); }} className="border border-red-500 px-2 py-1 hover:bg-red-900">RETRY</button>
          </div>
        )}

        {isConnected && (
          <div className="w-full h-full flex items-center justify-center">
            {aiSpeaking ? (
              <div className="flex gap-1 h-12 items-center">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="w-2 bg-green-500 animate-pulse" style={{ height: `${Math.random() * 100}%` }}></div>
                ))}
              </div>
            ) : (
              <div className="w-24 h-24 border border-green-500/30 rounded-full flex items-center justify-center animate-pulse">
                <div className="w-20 h-20 border border-green-500/50 rounded-full"></div>
              </div>
            )}
          </div>
        )}

        {/* Status Bar */}
        <div className="absolute bottom-0 left-0 right-0 bg-black border-t border-gray-800 px-2 py-1 flex justify-between items-center">
          <span className="text-[10px] text-green-500 font-mono uppercase">{aiSpeaking ? "INCOMING_AUDIO" : "LISTENING..."}</span>
          {isConnected && isMicOn && (
            <div className="flex gap-0.5 h-2 items-end">
              {[...Array(10)].map((_, i) => (
                <div key={i} className="w-1 bg-green-500" style={{ height: `${Math.max(10, Math.min(100, volumeLevel * 100 * Math.random()))}%`, opacity: volumeLevel > 0.01 ? 1 : 0.3 }} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className="p-3 bg-[#d4d4d4]">
        <div className="flex justify-between items-center mb-3 bg-white border-2 border-black px-2 py-1 shadow-retro-sm">
          <span className="font-mono text-xs font-bold">TIMER:</span>
          <span className="font-mono text-xs font-bold">{timerDisplay}</span>
        </div>

        <div className="flex justify-center gap-4">
          <button
            onClick={() => setIsMicOn(!isMicOn)}
            disabled={!isConnected}
            className={`w-10 h-10 flex items-center justify-center border-2 border-black shadow-retro-sm transition-all active:shadow-none active:translate-x-[2px] active:translate-y-[2px] ${isMicOn ? 'bg-white hover:bg-gray-100' : 'bg-red-500 text-white'
              }`}
          >
            {isMicOn ? <Mic size={18} /> : <MicOff size={18} />}
          </button>
          <button
            onClick={onClose}
            className="flex-1 bg-black text-white font-mono font-bold text-xs border-2 border-black hover:bg-gray-800 active:bg-gray-900 shadow-retro-sm active:shadow-none active:translate-x-[2px] active:translate-y-[2px] transition-all"
          >
            TERMINATE
          </button>
        </div>
      </div>
    </div>
  );
};

export default LiveInterviewPanel;
