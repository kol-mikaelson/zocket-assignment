from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import json
from datetime import datetime
import uvicorn
from enum import Enum
import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ad Text Rewriting Agent", version="1.0.0")

# Enums for validation
class ToneType(str, Enum):
    PROFESSIONAL = "professional"
    FUN = "fun"
    CASUAL = "casual"
    URGENT = "urgent"
    LUXURY = "luxury"
    FRIENDLY = "friendly"

class PlatformType(str, Enum):
    FACEBOOK = "facebook"
    GOOGLE = "google"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    TIKTOK = "tiktok"

# Request/Response Models
class AdRewriteRequest(BaseModel):
    original_text: str
    target_tone: ToneType
    target_platforms: List[PlatformType]
    brand_context: Optional[str] = None
    target_audience: Optional[str] = None

class PlatformOptimization(BaseModel):
    platform: PlatformType
    optimized_text: str
    character_count: int
    recommendations: List[str]
    performance_prediction: Dict[str, Union[float, Dict[str, float]]]

class AdRewriteResponse(BaseModel):
    request_id: str
    original_text: str
    tone_applied: ToneType
    platform_optimizations: List[PlatformOptimization]
    overall_insights: List[str]
    confidence_score: float
    timestamp: str

# Knowledge Graph Structure (Simplified)
class KnowledgeGraph:
    def __init__(self):
        self.platform_constraints = {
            PlatformType.FACEBOOK: {
                "max_chars": 125,
                "best_practices": ["Use emojis", "Include clear CTA", "Ask questions"],
                "tone_multipliers": {"fun": 1.2, "professional": 0.9, "casual": 1.1}
            },
            PlatformType.GOOGLE: {
                "max_chars": 90,
                "best_practices": ["Focus on keywords", "Clear value proposition", "Include location if relevant"],
                "tone_multipliers": {"professional": 1.3, "urgent": 1.1, "fun": 0.8}
            },
            PlatformType.INSTAGRAM: {
                "max_chars": 125,
                "best_practices": ["Use hashtags", "Visual storytelling", "Emojis"],
                "tone_multipliers": {"fun": 1.3, "casual": 1.2, "luxury": 1.1}
            },
            PlatformType.LINKEDIN: {
                "max_chars": 150,
                "best_practices": ["Professional language", "Industry insights", "Networking focus"],
                "tone_multipliers": {"professional": 1.4, "friendly": 1.1, "fun": 0.7}
            },
            PlatformType.TWITTER: {
                "max_chars": 280,
                "best_practices": ["Trending hashtags", "Concise messaging", "Engagement hooks"],
                "tone_multipliers": {"casual": 1.2, "urgent": 1.1, "fun": 1.0}
            },
            PlatformType.TIKTOK: {
                "max_chars": 100,
                "best_practices": ["Trending sounds", "Call-to-action", "Youth appeal"],
                "tone_multipliers": {"fun": 1.4, "casual": 1.3, "professional": 0.6}
            }
        }

        self.tone_patterns = {
            ToneType.PROFESSIONAL: {
                "keywords": ["expert", "solution", "proven", "results", "professional"],
                "style": "formal, authoritative, data-driven",
                "avoid": ["slang", "excessive emojis", "casual contractions"]
            },
            ToneType.FUN: {
                "keywords": ["awesome", "amazing", "exciting", "fun", "wow"],
                "style": "energetic, playful, emoji-rich",
                "avoid": ["formal language", "complex terms", "serious tone"]
            },
            ToneType.CASUAL: {
                "keywords": ["hey", "you", "easy", "simple", "great"],
                "style": "conversational, relatable, approachable",
                "avoid": ["overly formal", "complex jargon", "intimidating language"]
            },
            ToneType.URGENT: {
                "keywords": ["now", "limited", "hurry", "don't miss", "expires"],
                "style": "time-sensitive, action-oriented, compelling",
                "avoid": ["passive voice", "vague timelines", "weak CTAs"]
            },
            ToneType.LUXURY: {
                "keywords": ["exclusive", "premium", "elite", "sophisticated", "curated"],
                "style": "elegant, aspirational, high-quality",
                "avoid": ["cheap", "discount", "mass market terms"]
            },
            ToneType.FRIENDLY: {
                "keywords": ["welcome", "help", "together", "community", "support"],
                "style": "warm, supportive, inclusive",
                "avoid": ["aggressive sales", "pushy language", "cold corporate speak"]
            }
        }

# Initialize Knowledge Graph
kg = KnowledgeGraph()

# Memory Module for Pattern Recognition
class AgentMemory:
    def __init__(self):
        self.successful_rewrites = []
        self.performance_data = {}
        self.pattern_insights = {}

    def store_rewrite(self, request: AdRewriteRequest, response: AdRewriteResponse, feedback_score: Optional[float] = None):
        """Store successful rewrites for pattern learning"""
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "original_text": request.original_text,
            "tone": request.target_tone,
            "platforms": request.target_platforms,
            "confidence": response.confidence_score,
            "feedback_score": feedback_score
        }
        self.successful_rewrites.append(memory_entry)

        # Update pattern insights
        self._update_patterns(request, response)

    def _update_patterns(self, request: AdRewriteRequest, response: AdRewriteResponse):
        """Update pattern recognition based on successful rewrites"""
        key = f"{request.target_tone}_{len(request.target_platforms)}"
        if key not in self.pattern_insights:
            self.pattern_insights[key] = {"count": 0, "avg_confidence": 0}

        current = self.pattern_insights[key]
        current["count"] += 1
        current["avg_confidence"] = (current["avg_confidence"] * (current["count"] - 1) + response.confidence_score) / current["count"]

# Initialize Agent Memory
memory = AgentMemory()

# LLM Service using OpenAI client with OpenRouter
class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-4-maverick:free")
        self.site_url = os.getenv("SITE_URL", "http://localhost:8000")
        self.site_name = os.getenv("SITE_NAME", "Ad Rewriting Agent")

        # Initialize OpenAI client with OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    async def generate_rewrite(self, original_text: str, tone: str, platform: str,
                             brand_context: Optional[str] = None,
                             target_audience: Optional[str] = None,
                             char_limit: int = None) -> str:
        """Generate rewritten ad text using OpenAI client"""

        # Construct detailed prompt
        prompt = self._build_rewrite_prompt(
            original_text, tone, platform, brand_context, target_audience, char_limit
        )

        try:
            # Use OpenAI client with OpenRouter
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert marketing copywriter specializing in platform-optimized ad text. Create compelling, engaging copy that drives action while maintaining brand voice and platform best practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )

            generated_text = completion.choices[0].message.content

            # Clean up the response
            rewritten_text = self._clean_llm_response(generated_text, char_limit)
            return rewritten_text

        except Exception as e:
            print(f"LLM Error: {str(e)}")
            if "timeout" in str(e).lower():
                raise HTTPException(status_code=504, detail="LLM request timed out")
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                raise HTTPException(status_code=401, detail="Invalid API key or unauthorized access")
            elif "403" in str(e) or "forbidden" in str(e).lower():
                raise HTTPException(status_code=403, detail="Access forbidden - check model permissions")
            elif "429" in str(e) or "rate limit" in str(e).lower():
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            else:
                raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")

    def _build_rewrite_prompt(self, original_text: str, tone: str, platform: str,
                            brand_context: Optional[str], target_audience: Optional[str],
                            char_limit: Optional[int]) -> str:
        """Build comprehensive prompt for LLM rewriting"""

        tone_instructions = {
            "professional": "Use formal, authoritative language. Focus on expertise, results, and value propositions. Avoid casual language and excessive punctuation.",
            "fun": "Use energetic, playful language with emojis. Create excitement and enthusiasm. Use exclamation points and engaging questions.",
            "casual": "Use conversational, relatable language. Write like talking to a friend. Use contractions and simple, approachable terms.",
            "urgent": "Create time-sensitive urgency. Use action words like 'now', 'limited', 'hurry'. Include clear deadlines or scarcity indicators.",
            "luxury": "Use sophisticated, premium language. Emphasize exclusivity, quality, and prestige. Avoid discount or cheap terminology.",
            "friendly": "Use warm, welcoming language. Focus on community, support, and relationships. Create a sense of belonging."
        }

        platform_guidelines = {
            "facebook": f"Character limit: 125. Use engaging questions, emojis, and clear CTAs. Encourage comments and shares.",
            "google": f"Character limit: 90. Focus on keywords and clear value propositions. Include location if relevant.",
            "instagram": f"Character limit: 125. Use visual storytelling language and relevant hashtags. Appeal to aesthetics.",
            "linkedin": f"Character limit: 150. Use professional language and industry insights. Focus on business value.",
            "twitter": f"Character limit: 280. Use trending topics and hashtags. Create shareable, concise messages.",
            "tiktok": f"Character limit: 100. Use youth-oriented language and trends. Create viral, engaging content."
        }

        prompt = f"""
Rewrite the following ad text with these specifications:

ORIGINAL TEXT: "{original_text}"

REQUIREMENTS:
- Tone: {tone.upper()} - {tone_instructions.get(tone, 'Maintain appropriate tone')}
- Platform: {platform.upper()} - {platform_guidelines.get(platform, 'Follow platform best practices')}
- Character limit: {char_limit if char_limit else 'No strict limit'}
{f'- Brand context: {brand_context}' if brand_context else ''}
{f'- Target audience: {target_audience}' if target_audience else ''}

GUIDELINES:
1. Maintain the core message and intent of the original
2. Optimize for the specified platform's audience and format
3. Apply the requested tone consistently
4. Include a clear call-to-action
5. Stay within character limits
6. Make it compelling and action-oriented

Return ONLY the rewritten ad text, no explanations or quotes.
"""
        return prompt

    def _clean_llm_response(self, response: str, char_limit: Optional[int] = None) -> str:
        """Clean and format LLM response"""
        # Remove quotes if present
        cleaned = response.strip().strip('"').strip("'")

        # Remove any explanatory text that might be included
        lines = cleaned.split('\n')
        # Take the first substantial line as the ad text
        for line in lines:
            if line.strip() and len(line.strip()) > 10:
                cleaned = line.strip()
                break

        # Enforce character limit if specified
        if char_limit and len(cleaned) > char_limit:
            # Truncate at the last complete word within limit
            truncated = cleaned[:char_limit]
            last_space = truncated.rfind(' ')
            if last_space > char_limit * 0.8:  # Only truncate at word boundary if it's not too short
                cleaned = truncated[:last_space] + "..."
            else:
                cleaned = truncated + "..."

        return cleaned

    async def analyze_text_quality(self, original: str, rewritten: str, tone: str) -> Dict[str, float]:
        """Analyze the quality of rewritten text"""
        analysis_prompt = f"""
Analyze these two ad texts and provide scores from 0.0 to 1.0:

ORIGINAL: "{original}"
REWRITTEN: "{rewritten}"
TARGET TONE: {tone}

Rate the rewritten text on:
1. Tone consistency (how well it matches the target tone)
2. Message preservation (how well it maintains original intent)
3. Engagement potential (how likely it is to drive clicks/engagement)
4. Platform optimization (how well optimized for digital advertising)

Respond with ONLY a JSON object:
{{"tone_consistency": 0.0, "message_preservation": 0.0, "engagement_potential": 0.0, "platform_optimization": 0.0}}
"""

        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                },
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=100,
                temperature=0.3
            )

            analysis_text = completion.choices[0].message.content

            # Extract JSON from response
            try:
                scores = json.loads(analysis_text.strip())
                return scores
            except json.JSONDecodeError:
                # Fallback scores if JSON parsing fails
                return {
                    "tone_consistency": 0.7,
                    "message_preservation": 0.8,
                    "engagement_potential": 0.7,
                    "platform_optimization": 0.75
                }

        except Exception as e:
            print(f"Analysis error: {e}")

        # Default scores if analysis fails
        return {
            "tone_consistency": 0.7,
            "message_preservation": 0.8,
            "engagement_potential": 0.7,
            "platform_optimization": 0.75
        }

# Initialize LLM Service
try:
    llm_service = LLMService()
except ValueError as e:
    print(f"Warning: {e}")
    llm_service = None

# Core Agent Logic
class AdRewritingAgent:
    def __init__(self, knowledge_graph: KnowledgeGraph, memory: AgentMemory, llm_service: LLMService):
        self.kg = knowledge_graph
        self.memory = memory
        self.llm = llm_service

    async def rewrite_ad_text(self, request: AdRewriteRequest) -> AdRewriteResponse:
        """Main agent function to rewrite ad text using LLM"""
        if not self.llm:
            raise HTTPException(status_code=500, detail="LLM service not available - check API key configuration")

        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Generate platform optimizations using LLM
        platform_optimizations = []

        # Process each platform with LLM-powered rewriting
        for platform in request.target_platforms:
            platform_info = self.kg.platform_constraints[platform]
            char_limit = platform_info["max_chars"]

            # Generate optimized text using LLM
            optimized_text = await self.llm.generate_rewrite(
                original_text=request.original_text,
                tone=request.target_tone.value,
                platform=platform.value,
                brand_context=request.brand_context,
                target_audience=request.target_audience,
                char_limit=char_limit
            )

            # Create platform optimization object
            optimization = await self._create_platform_optimization(
                optimized_text, platform, request.target_tone, request.original_text
            )
            platform_optimizations.append(optimization)

        # Generate overall insights using LLM analysis
        insights = await self._generate_llm_insights(request, platform_optimizations)

        # Calculate confidence score based on LLM analysis
        confidence = await self._calculate_llm_confidence(request, platform_optimizations)

        response = AdRewriteResponse(
            request_id=request_id,
            original_text=request.original_text,
            tone_applied=request.target_tone,
            platform_optimizations=platform_optimizations,
            overall_insights=insights,
            confidence_score=confidence,
            timestamp=datetime.now().isoformat()
        )

        # Store in memory for pattern learning
        self.memory.store_rewrite(request, response)

        return response

    async def _create_platform_optimization(self, optimized_text: str, platform: PlatformType,
                                          tone: ToneType, original_text: str) -> PlatformOptimization:
        """Create platform optimization with LLM-powered analysis"""
        platform_info = self.kg.platform_constraints[platform]
        best_practices = platform_info["best_practices"]
        tone_multiplier = platform_info["tone_multipliers"].get(tone.value, 1.0)

        # Generate recommendations based on best practices
        recommendations = []
        for practice in best_practices:
            if practice.lower() not in optimized_text.lower():
                recommendations.append(f"Consider adding: {practice}")

        # Get quality analysis from LLM
        quality_scores = await self.llm.analyze_text_quality(original_text, optimized_text, tone.value)

        # Enhanced performance prediction using LLM quality scores
        base_ctr = 0.08
        base_engagement = 0.12
        base_conversion = 0.05

        # Apply quality multipliers
        quality_multiplier = (quality_scores["engagement_potential"] + quality_scores["platform_optimization"]) / 2

        performance_prediction = {
            "click_through_rate": min(0.15, tone_multiplier * base_ctr * quality_multiplier),
            "engagement_rate": min(0.25, tone_multiplier * base_engagement * quality_multiplier),
            "conversion_rate": min(0.10, tone_multiplier * base_conversion * quality_multiplier),
            "quality_scores": quality_scores
        }

        return PlatformOptimization(
            platform=platform,
            optimized_text=optimized_text,
            character_count=len(optimized_text),
            recommendations=recommendations,
            performance_prediction=performance_prediction
        )

    async def _generate_llm_insights(self, request: AdRewriteRequest, optimizations: List[PlatformOptimization]) -> List[str]:
        """Generate insights using LLM analysis"""
        insights = []

        # Analyze character usage across platforms
        char_counts = [opt.character_count for opt in optimizations]
        avg_chars = sum(char_counts) / len(char_counts)

        if avg_chars < 50:
            insights.append("Consider adding more descriptive content to increase engagement")
        elif avg_chars > 120:
            insights.append("Text might be too long for some platforms - consider more concise messaging")

        # Analyze tone-platform alignment
        problematic_combos = []
        if request.target_tone == ToneType.PROFESSIONAL and PlatformType.TIKTOK in request.target_platforms:
            problematic_combos.append("TikTok + Professional tone")
        if request.target_tone == ToneType.FUN and PlatformType.LINKEDIN in request.target_platforms:
            problematic_combos.append("LinkedIn + Fun tone")

        if problematic_combos:
            insights.append(f"Potential tone-platform misalignment: {', '.join(problematic_combos)}")

        # Quality-based insights
        avg_quality_scores = {}
        for opt in optimizations:
            quality_scores = opt.performance_prediction.get("quality_scores", {})
            for key, value in quality_scores.items():
                if key not in avg_quality_scores:
                    avg_quality_scores[key] = []
                avg_quality_scores[key].append(value)

        # Generate insights based on quality scores
        for metric, scores in avg_quality_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 0.6:
                if metric == "tone_consistency":
                    insights.append("Tone consistency could be improved across platforms")
                elif metric == "engagement_potential":
                    insights.append("Consider more engaging hooks and call-to-actions")
                elif metric == "platform_optimization":
                    insights.append("Platform-specific optimizations could be enhanced")

        # Performance prediction insights
        avg_ctr = sum(opt.performance_prediction["click_through_rate"] for opt in optimizations) / len(optimizations)
        if avg_ctr > 0.10:
            insights.append("High predicted performance - excellent tone-platform alignment")
        elif avg_ctr < 0.05:
            insights.append("Performance could be improved with better platform targeting")

        return insights[:5]  # Limit to top 5 insights

    async def _calculate_llm_confidence(self, request: AdRewriteRequest, optimizations: List[PlatformOptimization]) -> float:
        """Calculate confidence score using LLM quality analysis"""
        quality_scores = []

        for opt in optimizations:
            opt_quality = opt.performance_prediction.get("quality_scores", {})
            if opt_quality:
                # Average the quality metrics
                avg_quality = sum(opt_quality.values()) / len(opt_quality.values())
                quality_scores.append(avg_quality)

        if not quality_scores:
            return 0.7  # Default confidence

        base_confidence = sum(quality_scores) / len(quality_scores)

        # Adjust based on platform-tone alignment
        alignment_bonus = 0.0
        for opt in optimizations:
            platform_info = self.kg.platform_constraints[opt.platform]
            tone_multiplier = platform_info["tone_multipliers"].get(request.target_tone.value, 1.0)
            if tone_multiplier > 1.0:
                alignment_bonus += 0.02

        # Consider memory patterns
        pattern_key = f"{request.target_tone}_{len(request.target_platforms)}"
        pattern_bonus = 0.0
        if pattern_key in self.memory.pattern_insights:
            pattern_confidence = self.memory.pattern_insights[pattern_key]["avg_confidence"]
            pattern_bonus = (pattern_confidence - 0.7) * 0.1  # Small adjustment based on history

        final_confidence = min(0.95, base_confidence + alignment_bonus + pattern_bonus)
        return max(0.1, final_confidence)  # Ensure minimum confidence

# Initialize Agent (with error handling)
if llm_service:
    agent = AdRewritingAgent(kg, memory, llm_service)
else:
    agent = None

# API Routes
@app.post("/run-agent", response_model=AdRewriteResponse)
async def run_agent(request: AdRewriteRequest):
    """Main agent endpoint to rewrite ad text using LLM"""
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized - check environment variables")

    try:
        response = await agent.rewrite_ad_text(request)
        return response
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        import traceback
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_service_available": llm_service is not None,
        "agent_available": agent is not None
    }

@app.get("/agent-stats")
async def get_agent_stats():
    """Get agent performance statistics"""
    return {
        "total_rewrites": len(memory.successful_rewrites),
        "pattern_insights": memory.pattern_insights,
        "supported_tones": [tone.value for tone in ToneType],
        "supported_platforms": [platform.value for platform in PlatformType],
        "llm_model": llm_service.model if llm_service else None,
        "service_status": {
            "llm_available": llm_service is not None,
            "agent_available": agent is not None
        }
    }

@app.post("/feedback")
async def submit_feedback(request_id: str, feedback_score: float):
    """Submit feedback for a specific rewrite (for improvement loop)"""
    # In production, this would update the memory with feedback
    return {"message": "Feedback received", "request_id": request_id, "score": feedback_score}

@app.get("/example")
async def get_example():
    """Get an example request for testing"""
    return {
        "example_request": {
            "original_text": "Buy our product now! Great deals available!",
            "target_tone": "professional",
            "target_platforms": ["facebook", "linkedin", "google"],
            "brand_context": "B2B software company",
            "target_audience": "Marketing professionals"
        }
    }

@app.post("/test-llm")
async def test_llm_connection():
    """Test LLM connectivity"""
    if not llm_service:
        return {
            "status": "error",
            "error": "LLM service not initialized - check OPENROUTER_API_KEY"
        }

    try:
        test_response = await llm_service.generate_rewrite(
            original_text="Test ad copy for connectivity check",
            tone="professional",
            platform="facebook",
            char_limit=125
        )
        return {
            "status": "success",
            "llm_model": llm_service.model,
            "test_response": test_response
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
