import { Button } from "@/components/ui/button";
import { Sprout, Leaf, TreePine, Wheat } from "lucide-react";

interface EmptyStateProps {
  onExampleClick: (message: string) => void;
}

export const EmptyState = ({ onExampleClick }: EmptyStateProps) => {
  const examples = [
    {
      icon: Sprout,
      title: "Crop Disease Detection",
      description: "How can I identify early signs of blight in tomatoes?",
    },
    {
      icon: Leaf,
      title: "Soil Management",
      description: "What are the best practices for improving soil fertility?",
    },
    {
      icon: TreePine,
      title: "Sustainable Farming",
      description: "Explain organic farming methods for small-scale farmers",
    },
    {
      icon: Wheat,
      title: "Crop Planning",
      description: "When is the best time to plant wheat in northern regions?",
    },
  ];

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="max-w-2xl text-center space-y-8">
        {/* Logo and Title */}
        <div className="space-y-4">
          <div className="h-16 w-16 mx-auto rounded-full bg-gradient-primary flex items-center justify-center">
            <Sprout className="h-8 w-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground mb-2">
              Welcome to Krishi-RAG
            </h1>
            <p className="text-lg text-muted-foreground">
              Your AI-powered agricultural intelligence assistant. Ask me anything about farming, crops, soil management, and sustainable agriculture.
            </p>
          </div>
        </div>

        {/* Example Questions */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {examples.map((example, index) => {
            const Icon = example.icon;
            return (
              <Button
                key={index}
                variant="outline"
                className="h-auto p-4 flex flex-col items-start space-y-2 hover:bg-accent transition-colors"
                onClick={() => onExampleClick(example.description)}
              >
                <div className="flex items-center gap-2 w-full">
                  <Icon className="h-5 w-5 text-emerald shrink-0" />
                  <span className="font-medium text-sm text-left">{example.title}</span>
                </div>
                <p className="text-xs text-muted-foreground text-left">
                  {example.description}
                </p>
              </Button>
            );
          })}
        </div>

        {/* Features */}
        <div className="text-sm text-muted-foreground space-y-2">
          <p className="font-medium">✨ Powered by advanced agricultural knowledge</p>
          <div className="flex flex-wrap justify-center gap-4 text-xs">
            <span>🌱 Crop Management</span>
            <span>🚜 Farming Techniques</span>
            <span>🌾 Harvest Optimization</span>
            <span>💧 Water Management</span>
          </div>
        </div>
      </div>
    </div>
  );
};