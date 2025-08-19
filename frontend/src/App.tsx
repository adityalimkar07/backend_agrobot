import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "@/hooks/useAuth";
import { useEffect, useState } from "react";
import Index from "./pages/Index";
import Auth from "./pages/Auth";
import Privacy from "./pages/Privacy";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => {
  const [locationStatus, setLocationStatus] = useState<'requesting' | 'granted' | 'denied' | 'unsupported'>('requesting');
  const [locationInfo, setLocationInfo] = useState<any>(null);

  // Location request functions
  const requestUserLocation = async () => {
    try {
      if (!navigator.geolocation) {
        setLocationStatus('unsupported');
        console.log('Geolocation not supported, using IP-based detection');
        // Fallback to IP-based location
        await fetch('/location/auto', { method: 'POST' });
        await updateLocationInfo();
        return;
      }

      const position = await new Promise<GeolocationPosition>((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          resolve,
          reject,
          {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 600000 // 10 minutes
          }
        );
      });

      const { latitude, longitude } = position.coords;
      
      // Send coordinates to backend
      const response = await fetch('/location/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude, longitude })
      });

      if (response.ok) {
        setLocationStatus('granted');
        console.log(`📍 Location set: ${latitude}, ${longitude}`);
        await updateLocationInfo();
      } else {
        throw new Error('Failed to update backend location');
      }

    } catch (error) {
      setLocationStatus('denied');
      console.log('Location permission denied or failed, using IP-based detection');
      // Fallback to IP-based location
      try {
        await fetch('/location/auto', { method: 'POST' });
        await updateLocationInfo();
      } catch (ipError) {
        console.error('IP-based location also failed:', ipError);
      }
    }
  };

  // Update location info from backend
  const updateLocationInfo = async () => {
    try {
      const response = await fetch('/location/current');
      const data = await response.json();
      setLocationInfo(data);
    } catch (error) {
      console.error('Failed to get location status:', error);
    }
  };

  // Request location on app startup
  useEffect(() => {
    console.log('🌍 App started, requesting user location...');
    requestUserLocation();
  }, []);

  // Optional: Log location status changes
  useEffect(() => {
    if (locationStatus !== 'requesting') {
      console.log(`📍 Location status: ${locationStatus}`);
    }
  }, [locationStatus]);

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          {/* Optional: Show location status indicator */}
          {locationInfo && (
            <div className="fixed top-4 right-4 z-50 bg-white shadow-lg rounded-lg p-3 border">
              <div className="text-sm">
                <div className="font-medium">
                  📍 Location: {locationInfo.success ? 'Detected' : 'Default'}
                </div>
                <div className="text-gray-600">
                  {locationInfo.coordinates.latitude.toFixed(2)}, {locationInfo.coordinates.longitude.toFixed(2)}
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {locationInfo.message}
                </div>
              </div>
            </div>
          )}
          
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/auth" element={<Auth />} />
              <Route path="/privacy" element={<Privacy />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
};

export default App;