import { useState, useRef, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from './use-toast';

export interface AudioRecorderState {
  isRecording: boolean;
  isProcessing: boolean;
  audioURL: string | null;
}

export const useAudioRecorder = () => {
  const [state, setState] = useState<AudioRecorderState>({
    isRecording: false,
    isProcessing: false,
    audioURL: null,
  });
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const { toast } = useToast();

  const startRecording = useCallback(async () => {
    try {
      console.log('Starting audio recording...');
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log('Recording stopped');
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const audioURL = URL.createObjectURL(audioBlob);
        setState(prev => ({ ...prev, audioURL }));
      };

      mediaRecorder.start();
      setState(prev => ({ ...prev, isRecording: true }));
      
      console.log('Recording started successfully');
      
    } catch (error) {
      console.error('Error starting recording:', error);
      toast({
        title: "Error",
        description: "Could not access microphone. Please check permissions.",
        variant: "destructive",
      });
    }
  }, [toast]);

  const stopRecording = useCallback(() => {
    console.log('Stopping recording...');
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    setState(prev => ({ ...prev, isRecording: false }));
  }, []);

  const transcribeAudio = useCallback(async (language: string = 'auto'): Promise<string> => {
    if (!chunksRef.current.length) {
      throw new Error('No audio data to transcribe');
    }

    setState(prev => ({ ...prev, isProcessing: true }));

    try {
      console.log('Converting audio to base64...');
      
      // Create audio blob
      const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
      
      // Convert to base64
      const reader = new FileReader();
      const base64Promise = new Promise<string>((resolve, reject) => {
        reader.onload = () => {
          const result = reader.result as string;
          // Remove the data URL prefix to get just the base64 data
          const base64 = result.split(',')[1];
          resolve(base64);
        };
        reader.onerror = reject;
      });
      
      reader.readAsDataURL(audioBlob);
      const base64Audio = await base64Promise;
      
      console.log('Sending audio to transcription service...');
      
      // Call our Supabase edge function
      const { data, error } = await supabase.functions.invoke('speech-to-text', {
        body: { 
          audio: base64Audio,
          language: language === 'auto' ? undefined : language
        }
      });

      if (error) {
        console.error('Supabase function error:', error);
        throw new Error(error.message || 'Transcription service error');
      }

      if (data.error) {
        console.error('Transcription error:', data.error);
        throw new Error(data.error);
      }

      console.log('Transcription successful:', data.text);
      return data.text;

    } catch (error) {
      console.error('Error transcribing audio:', error);
      throw error;
    } finally {
      setState(prev => ({ ...prev, isProcessing: false }));
    }
  }, []);

  const clearRecording = useCallback(() => {
    console.log('Clearing recording...');
    chunksRef.current = [];
    if (state.audioURL) {
      URL.revokeObjectURL(state.audioURL);
    }
    setState({
      isRecording: false,
      isProcessing: false,
      audioURL: null,
    });
  }, [state.audioURL]);

  return {
    ...state,
    startRecording,
    stopRecording,
    transcribeAudio,
    clearRecording,
  };
};