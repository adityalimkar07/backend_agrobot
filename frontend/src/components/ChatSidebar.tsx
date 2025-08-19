import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Shield, X, LogOut, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/useAuth";
import { useConversations, type Conversation } from "@/hooks/useConversations";
import { formatDistanceToNow } from "date-fns";
import { Link, useNavigate } from "react-router-dom";

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  className?: string;
}

export const ChatSidebar = ({ isOpen, onClose, className }: ChatSidebarProps) => {
  const { user, signOut } = useAuth();
  const { conversations, currentConversation, selectConversation, deleteConversation, startNewConversation } = useConversations();
  const navigate = useNavigate();

  const handleConversationClick = (conversation: Conversation) => {
    selectConversation(conversation);
    onClose();
    navigate('/'); // Navigate back to main chat page to show the conversation
  };

  const handleNewChat = () => {
    startNewConversation();
    onClose();
    navigate('/'); // Navigate back to main chat page
  };

  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation();
    deleteConversation(conversationId);
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <aside className={cn(
        "fixed left-0 top-0 h-full w-80 bg-background border-r z-50 transform transition-transform duration-200 ease-in-out",
        "lg:relative lg:transform-none lg:z-auto lg:w-80",
        "md:w-72 sm:w-64",
        isOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        className
      )}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 rounded bg-gradient-primary flex items-center justify-center">
                <span className="text-white font-bold text-xs">K</span>
              </div>
              <span className="font-semibold text-sm">Krishi-RAG</span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="lg:hidden h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          {/* Chat History */}
          <div className="flex-1 overflow-hidden">
            <div className="p-4 flex items-center justify-between">
              <h3 className="text-sm font-medium text-muted-foreground">Recent Chats</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleNewChat}
                className="h-7 px-2 text-xs"
              >
                New Chat
              </Button>
            </div>
            <ScrollArea className="flex-1 px-4">
              <div className="space-y-2 pb-4">
                {conversations.map((conversation) => (
                  <div
                    key={conversation.id}
                    className={cn(
                      "group relative",
                      currentConversation?.id === conversation.id && "bg-accent/50 rounded-lg"
                    )}
                  >
                    <Button
                      variant="ghost"
                      className="w-full justify-start h-auto p-3 text-left hover:bg-accent pr-8"
                      onClick={() => handleConversationClick(conversation)}
                    >
                      <MessageSquare className="h-4 w-4 mr-3 shrink-0 text-muted-foreground" />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium truncate">{conversation.title}</div>
                        <div className="text-xs text-muted-foreground">
                          {formatDistanceToNow(new Date(conversation.updated_at), { addSuffix: true })}
                        </div>
                      </div>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="absolute right-1 top-1/2 -translate-y-1/2 h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                      onClick={(e) => handleDeleteConversation(e, conversation.id)}
                    >
                      <Trash2 className="h-3 w-3 text-destructive" />
                    </Button>
                  </div>
                ))}
                {conversations.length === 0 && (
                  <div className="text-center text-sm text-muted-foreground py-4">
                    No conversations yet.<br />Start chatting to see your history!
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Footer */}
          <div className="border-t p-4 space-y-2">
            {user && (
              <div className="text-xs text-muted-foreground mb-2 px-3">
                Signed in as {user.email}
              </div>
            )}
            <Link to="/privacy">
              <Button variant="ghost" className="w-full justify-start gap-3">
                <Shield className="h-4 w-4" />
                Privacy Policy
              </Button>
            </Link>
            {user && (
              <Button 
                variant="ghost" 
                className="w-full justify-start gap-3 text-destructive hover:text-destructive"
                onClick={signOut}
              >
                <LogOut className="h-4 w-4" />
                Sign Out
              </Button>
            )}
          </div>
        </div>
      </aside>
    </>
  );
};