import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

// Process base64 in chunks to prevent memory issues
function processBase64Chunks(base64String: string, chunkSize = 32768) {
  const chunks: Uint8Array[] = [];
  let position = 0;
  
  while (position < base64String.length) {
    const chunk = base64String.slice(position, position + chunkSize);
    const binaryChunk = atob(chunk);
    const bytes = new Uint8Array(binaryChunk.length);
    
    for (let i = 0; i < binaryChunk.length; i++) {
      bytes[i] = binaryChunk.charCodeAt(i);
    }
    
    chunks.push(bytes);
    position += chunkSize;
  }

  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result;
}

serve(async (req) => {
  console.log('Speech-to-text function called');
  
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const { audio, language = 'auto' } = await req.json();
    
    if (!audio) {
      throw new Error('No audio data provided');
    }

    console.log('Processing audio data...');
    
    // Process audio in chunks to prevent memory issues
    const binaryAudio = processBase64Chunks(audio);
    
    console.log(`Audio data processed, size: ${binaryAudio.length} bytes`);
    
    // Convert binary audio to base64 for Google API
    const base64Audio = btoa(String.fromCharCode(...binaryAudio));
    
    // Map language codes for Google Speech-to-Text
    let languageCode = 'hi-IN'; // Default to Hindi
    if (language === 'en' || language === 'english') {
      languageCode = 'en-IN'; // English (India) for better Hinglish support
    } else if (language === 'hi' || language === 'hindi') {
      languageCode = 'hi-IN';
    }

    // Prepare request body for Google Speech-to-Text v2
    const requestBody = {
      config: {
        encoding: 'WEBM_OPUS',
        sampleRateHertz: 16000,
        languageCode: languageCode,
        alternativeLanguageCodes: ['en-IN', 'hi-IN'], // Support for Hinglish
        enableAutomaticPunctuation: true,
        enableWordTimeOffsets: false,
        model: 'latest_long'
      },
      audio: {
        content: base64Audio
      }
    };

    console.log('Sending request to Google Speech-to-Text API...');
    
    // Send to Google Speech-to-Text API v2
    const response = await fetch(`https://speech.googleapis.com/v1/speech:recognize?key=${Deno.env.get('GOOGLE_CLOUD_API_KEY')}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Google Speech-to-Text API error:', errorText);
      throw new Error(`Google Speech-to-Text API error: ${errorText}`);
    }

    const result = await response.json();
    console.log('Google Speech-to-Text response:', result);

    if (!result.results || result.results.length === 0) {
      throw new Error('No transcription results from Google Speech-to-Text');
    }

    const transcription = result.results[0]?.alternatives[0]?.transcript || '';
    const detectedLanguage = result.results[0]?.languageCode || languageCode;
    
    console.log('Transcription successful:', transcription);

    return new Response(
      JSON.stringify({ 
        text: transcription,
        language: detectedLanguage
      }),
      { 
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        } 
      }
    );

  } catch (error) {
    console.error('Error in speech-to-text function:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message || 'Unknown error occurred' 
      }),
      {
        status: 500,
        headers: { 
          ...corsHeaders, 
          'Content-Type': 'application/json' 
        },
      }
    );
  }
});