import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Loader2, Mic, MicOff, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { useToast } from "@/hooks/use-toast";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export const ChatInput = ({ onSendMessage, isLoading = false, disabled = false }: ChatInputProps) => {
  const [message, setMessage] = useState("");
  const { toast } = useToast();
  const { 
    isRecording, 
    isProcessing, 
    startRecording, 
    stopRecording, 
    transcribeAudio, 
    clearRecording 
  } = useAudioRecorder();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSendMessage(message.trim());
      setMessage("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleAudioTranscription = async (language: string = 'auto') => {
    try {
      const transcriptedText = await transcribeAudio(language);
      if (transcriptedText.trim()) {
        setMessage(prev => prev ? `${prev} ${transcriptedText}` : transcriptedText);
        toast({
          title: "Audio Transcribed",
          description: `Text converted successfully in ${language === 'auto' ? 'auto-detected language' : language}`,
        });
      }
      clearRecording();
    } catch (error) {
      console.error('Transcription error:', error);
      toast({
        title: "Transcription Failed",
        description: error instanceof Error ? error.message : "Could not convert audio to text",
        variant: "destructive",
      });
      clearRecording();
    }
  };

  const handleMicClick = async () => {
    if (isRecording) {
      stopRecording();
      // After stopping, show language selection dropdown
      setTimeout(() => {
        // This will be handled by the dropdown menu
      }, 100);
    } else {
      await startRecording();
    }
  };

  return (
    <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <form onSubmit={handleSubmit} className="flex gap-2 p-4">
        <div className="flex-1 relative">
          <Textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Krishi-RAG anything about agriculture..."
            className={cn(
              "min-h-[44px] max-h-32 resize-none pr-12 bg-input border-border",
              "focus:ring-2 focus:ring-ring focus:border-transparent"
            )}
            disabled={disabled || isLoading}
          />
        </div>
        
        {/* Audio Recording Button */}
        {!isRecording ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                type="button"
                size="icon"
                variant="outline"
                disabled={disabled || isLoading || isProcessing}
                className={cn(
                  "h-11 w-11 shrink-0",
                  isProcessing && "opacity-50 cursor-not-allowed"
                )}
              >
                {isProcessing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Mic className="h-4 w-4" />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => startRecording()}>
                <Mic className="h-4 w-4 mr-2" />
                Start Recording
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                type="button"
                size="icon"
                variant="destructive"
                className="h-11 w-11 shrink-0 animate-pulse"
              >
                <Square className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem onClick={() => { stopRecording(); setTimeout(() => handleAudioTranscription('auto'), 200); }}>
                <Mic className="h-4 w-4 mr-2" />
                Auto Language
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => { stopRecording(); setTimeout(() => handleAudioTranscription('en'), 200); }}>
                <Mic className="h-4 w-4 mr-2" />
                English
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => { stopRecording(); setTimeout(() => handleAudioTranscription('hi'), 200); }}>
                <Mic className="h-4 w-4 mr-2" />
                Hindi
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => { stopRecording(); clearRecording(); }}>
                <MicOff className="h-4 w-4 mr-2" />
                Cancel
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        
        <Button
          type="submit"
          size="icon"
          disabled={!message.trim() || isLoading || disabled}
          className={cn(
            "h-11 w-11 shrink-0 bg-primary hover:bg-primary/90",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </form>
    </div>
  );
};