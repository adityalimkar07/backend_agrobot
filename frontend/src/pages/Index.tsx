import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ChatMessage, type Message } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { ChatHeader } from "@/components/ChatHeader";
import { ChatSidebar } from "@/components/ChatSidebar";
import { EmptyState } from "@/components/EmptyState";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/useAuth";
import { useConversations } from "@/hooks/useConversations";

type AskResponse = {
  answer: string;
  query_analysis?: Record<string, unknown>;
  performance?: Record<string, unknown>;
  economics?: Record<string, unknown>;
  weather_context_provided?: boolean;
  multilingual_info?: Record<string, unknown>;
  retrieved_documents?: unknown[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const { user, loading: authLoading } = useAuth();
  const {
    messages,
    currentConversation,
    createConversation,
    addMessage,
    startNewConversation,
    selectConversation,
  } = useConversations();
  const navigate = useNavigate();

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      ) as HTMLDivElement | null;
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  // Redirect to auth if not authenticated
  useEffect(() => {
    if (!authLoading && !user) {
      navigate("/auth");
    }
  }, [user, authLoading, navigate]);

  const handleSendMessage = async (content: string) => {
    if (!user || !content.trim()) return;

    setIsLoading(true);

    try {
      // Ensure a conversation exists
      let conversation = currentConversation;
      if (!conversation) {
        const title =
          content.length > 50 ? content.substring(0, 50) + "..." : content;
        conversation = await createConversation(title);
        if (!conversation) {
          throw new Error("Failed to create conversation");
        }
      }

      // Add user message instantly
      await addMessage(conversation.id, content, "user");

      // Call your local backend
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: content }),
      });

      if (!res.ok) {
        // Try to surface server error text
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status} ${res.statusText} ${text || ""}`);
      }

      const data = (await res.json()) as AskResponse;
      const aiResponse =
        (data?.answer && String(data.answer).trim()) ||
        "Sorry, I couldn't generate an answer.";

      // Add assistant message
      await addMessage(conversation.id, aiResponse, "assistant");
    } catch (error: any) {
      console.error(error);
      toast({
        title: "Request failed",
        description:
          error?.message ||
          "Failed to reach the backend. Is it running on http://127.0.0.1:8000?",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewChat = () => {
    startNewConversation();
    toast({
      title: "New Chat Started",
      description: "Ready for your agricultural questions!",
    });
  };

  const handleExampleClick = (message: string) => {
    handleSendMessage(message);
  };

  // Show loading state while checking auth
  if (authLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="h-8 w-8 rounded bg-gradient-primary flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold">K</span>
          </div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Show sign in prompt if not authenticated
  if (!user) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="h-8 w-8 rounded bg-gradient-primary flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold">K</span>
          </div>
          <h1 className="text-2xl font-bold mb-2">Welcome to Krishi-RAG</h1>
          <p className="text-muted-foreground mb-4">
            Please sign in to access your agricultural AI assistant
          </p>
          <Button onClick={() => navigate("/auth")}>Sign In</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-background">
      {/* Sidebar */}
      <ChatSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        className="hidden lg:block"
      />

      {/* Mobile Sidebar */}
      <ChatSidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        className="lg:hidden"
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatHeader
          onNewChat={handleNewChat}
          onToggleSidebar={() => setSidebarOpen(true)}
        />

        <div className="flex-1 flex flex-col min-h-0">
          {messages.length === 0 ? (
            <EmptyState onExampleClick={handleExampleClick} />
          ) : (
            <ScrollArea ref={scrollAreaRef} className="flex-1">
              <div className="max-w-4xl mx-auto">
                {messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={{
                      id: message.id,
                      content: message.content,
                      role: message.role,
                      timestamp: new Date(message.created_at),
                    }}
                  />
                ))}
                {isLoading && (
                  <div className="flex justify-start p-4">
                    <div className="bg-ai-bubble rounded-2xl px-4 py-3 max-w-[80%]">
                      <div className="flex items-center gap-2 text-ai-bubble-foreground">
                        <div className="flex gap-1">
                          <div className="w-2 h-2 rounded-full animate-pulse bg-current"></div>
                          <div
                            className="w-2 h-2 rounded-full animate-pulse bg-current"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                          <div
                            className="w-2 h-2 rounded-full animate-pulse bg-current"
                            style={{ animationDelay: "0.4s" }}
                          ></div>
                        </div>
                        <span className="text-sm">Thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          )}
        </div>

        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default Index;
