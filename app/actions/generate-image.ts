"use server"

import OpenAI from "openai"
import Replicate from "replicate"
// Note: We dynamically import FormData and node-fetch when needed
// This is to avoid issues with Next.js server components

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY as string,
})

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN as string,
})

// Available aspect ratios with their corresponding dimensions
const aspectRatios = {
  "1:1": { width: 1024, height: 1024, size: "1024x1024" },
  "16:9": { width: 1536, height: 864, size: "1536x1024" },
  "9:16": { width: 864, height: 1536, size: "1024x1536" },
  "3:4": { width: 896, height: 1152, size: "1024x1536" },
  "4:3": { width: 1152, height: 896, size: "1536x1024" },
  "2:3": { width: 832, height: 1216, size: "1024x1536" },
  "3:2": { width: 1216, height: 832, size: "1536x1024" }
};

// Export the AspectRatio type so it can be imported in the client component
export type AspectRatio = keyof typeof aspectRatios;

/**
 * Generate multiple images in parallel using the OpenAI GPT-Image-1 model.
 * This function accepts a text prompt and an aspect ratio, returning up to 4 generated images.
 */
export async function generateImages(
  prompt: string,
  aspectRatio: AspectRatio = "1:1",
  numberOfImages: number = 4
): Promise<{ 
  success: boolean 
  images?: string[] 
  error?: string 
}> {
  try {
    if (!process.env.OPENAI_API_KEY) {
      throw new Error("OPENAI_API_KEY is not configured")
    }

    console.log(`Starting image generation with prompt: "${prompt}" and aspect ratio: ${aspectRatio}`)
    
    // Get the size parameter based on the aspect ratio
    const size = aspectRatios[aspectRatio].size
    
    // Generate predictions for the specified number of images (up to 4)
    // TypeScript definitions don't match actual gpt-image-1 supported parameters
    const result = await openai.images.generate({
      model: "gpt-image-1",
      prompt: prompt,
      n: Math.min(numberOfImages, 4),
      size: size as any, // gpt-image-1 supports different sizes than the type definition
      quality: "high" as any, // gpt-image-1 supports "high" quality
      // Note: We're using b64_json by default which is what GPT-Image-1 returns
    } as any)
    
    console.log("Generation complete, received results")
    
    // Extract base64 image data and convert to data URLs
    const imageUrls = result.data.map(image => {
      if (image.b64_json) {
        return `data:image/png;base64,${image.b64_json}`
      }
      return ""
    }).filter(Boolean)

    if (imageUrls.length === 0) {
      throw new Error("No images were generated")
    }

    const response = {
      success: true,
      images: imageUrls,
    }

    // Ensure the response is serializable
    return JSON.parse(JSON.stringify(response))
  } catch (error) {
    console.error("Error in generateImages:", error)
    return {
      success: false,
      error: error instanceof Error ? String(error.message) : "Failed to generate images",
    }
  }
}

/**
 * Inpaint an image using the 'ideogram-ai/ideogram-v2' model.
 * This function expects:
 *  - `prompt` (string)
 *  - `base64Image` (the base64 data URL of the original image)
 *  - `maskImage` (the base64 data URL of the mask, where black = inpaint region, white = keep)
 */
export async function inpaintImage(
  prompt: string,
  base64Image: string,
  maskImage: string
): Promise<{
  success: boolean
  image?: string
  error?: string
}> {
  try {
    if (!process.env.REPLICATE_API_TOKEN) {
      throw new Error("REPLICATE_API_TOKEN is not configured")
    }

    console.log("Starting inpainting with prompt:", prompt)

    // Verify we have valid data URLs
    if (!base64Image.startsWith('data:image/')) {
      throw new Error("Invalid image format: must be a data URL")
    }
    if (!maskImage.startsWith('data:image/')) {
      throw new Error("Invalid mask format: must be a data URL")
    }

    // The safer way is to upload these images first, then use the URLs
    try {
      // Create the prediction with the full prediction API
      const prediction = await replicate.predictions.create({
        version: "ideogram-ai/ideogram-v2",
        input: {
          prompt: prompt,
          image: base64Image,
          mask: maskImage,
          negative_prompt: "low quality, bad anatomy, blurry, pixelated",
          style_type: "General"
        },
      });

      console.log("Prediction ID:", prediction.id);
      
      // Poll for the prediction result
      let completedPrediction = prediction;
      let attempts = 0;
      const maxAttempts = 30; // Stop after 30 attempts (30 seconds)
      
      while (attempts < maxAttempts) {
        // Get the latest prediction
        completedPrediction = await replicate.predictions.get(prediction.id);
        console.log(`Polling attempt ${attempts + 1}/${maxAttempts}: ${completedPrediction.status}`);

        if (completedPrediction.status === "succeeded") {
          break;
        } else if (completedPrediction.status === "failed" || completedPrediction.status === "canceled") {
          throw new Error(`Prediction failed with status: ${completedPrediction.status}`);
        }

        // Wait before the next polling attempt
        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
      }

      if (attempts >= maxAttempts) {
        throw new Error("Prediction timed out after 30 seconds");
      }

      // Process the output
      console.log("Prediction succeeded with output:", completedPrediction.output);
      
      const output = completedPrediction.output;
      
      // Handle different output formats
      let imageUrl: string | null = null;
      
      if (typeof output === "string") {
        imageUrl = output;
      } else if (Array.isArray(output) && output.length > 0) {
        imageUrl = output[0];
      }

      if (!imageUrl) {
        throw new Error("No image URL in prediction output");
      }

      return {
        success: true,
        image: imageUrl,
      };
    } catch (apiError: any) {
      console.error("API Error details:", apiError);
      
      // Extract more detailed error message if available
      let errorMessage = "Unknown API error";
      
      if (apiError.message) {
        errorMessage = apiError.message;
      } else if (apiError.response && apiError.response.data) {
        errorMessage = JSON.stringify(apiError.response.data);
      }
      
      throw new Error(`API Error: ${errorMessage}`);
    }
  } catch (error) {
    console.error("Error in inpaintImage:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Inpainting failed",
    };
  }
}