import { Button } from "@/components/ui/button";
import { PlusCircle, Menu } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatHeaderProps {
  onNewChat?: () => void;
  onToggleSidebar?: () => void;
  className?: string;
}

export const ChatHeader = ({ onNewChat, onToggleSidebar, className }: ChatHeaderProps) => {
  return (
    <header className={cn(
      "flex items-center justify-between p-4 border-b bg-gradient-subtle",
      className
    )}>
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleSidebar}
          className="lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </Button>
        
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-primary flex items-center justify-center">
            <span className="text-white font-bold text-sm">K</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-foreground">Krishi-RAG</h1>
            <p className="text-xs text-muted-foreground">Agricultural Intelligence Assistant</p>
          </div>
        </div>
      </div>

      <Button
        variant="outline"
        size="sm"
        onClick={onNewChat}
        className="gap-2 hover:bg-accent"
      >
        <PlusCircle className="h-4 w-4" />
        <span className="hidden sm:inline">New Chat</span>
      </Button>
    </header>
  );
};