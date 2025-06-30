import requests
import json
import asyncio
from typing import Dict, Any

class AdRewritingAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def test_agent(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the ad rewriting agent with sample data"""
        try:
            response = requests.post(f"{self.base_url}/run-agent", json=request_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def test_llm_connection(self) -> Dict[str, Any]:
        """Test LLM connectivity"""
        try:
            response = requests.post(f"{self.base_url}/test-llm")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        try:
            response = requests.get(f"{self.base_url}/agent-stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the agent is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

def run_comprehensive_test():
    """Run comprehensive tests of the ad rewriting agent"""
    client = AdRewritingAgentClient()
    
    print("🤖 Ad Text Rewriting Agent - Test Suite")
    print("=" * 50)
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Test cases
    test_cases = [
        {
            "name": "Professional B2B Ad",
            "request": {
                "original_text": "Buy our software now! Amazing deals this month!",
                "target_tone": "professional",
                "target_platforms": ["linkedin", "google"],
                "brand_context": "Enterprise software company",
                "target_audience": "IT decision makers"
            }
        },
        {
            "name": "Fun Social Media Ad",
            "request": {
                "original_text": "Check out our new product line",
                "target_tone": "fun",
                "target_platforms": ["instagram", "tiktok", "facebook"],
                "brand_context": "Fashion brand for young adults",
                "target_audience": "Gen Z consumers"
            }
        },
        {
            "name": "Urgent Limited-Time Offer",
            "request": {
                "original_text": "Save money on our premium service",
                "target_tone": "urgent",
                "target_platforms": ["facebook", "google", "twitter"],
                "brand_context": "SaaS subscription service",
                "target_audience": "Small business owners"
            }
        },
        {
            "name": "Luxury Brand Positioning",
            "request": {
                "original_text": "High quality products at great prices",
                "target_tone": "luxury",
                "target_platforms": ["instagram", "linkedin"],
                "brand_context": "Premium lifestyle brand",
                "target_audience": "High-income professionals"
            }
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 2):
        print(f"\n{i}. {test_case['name']}:")
        print("-" * 30)
        
        result = client.test_agent(test_case["request"])
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"Original: {result['original_text']}")
        print(f"Tone Applied: {result['tone_applied']}")
        print(f"Confidence Score: {result['confidence_score']:.2f}")
        
        print("\nPlatform Optimizations:")
        for opt in result["platform_optimizations"]:
            print(f"  📱 {opt['platform'].upper()}:")
            print(f"    Text: {opt['optimized_text']}")
            print(f"    Characters: {opt['character_count']}")
            print(f"    Predicted CTR: {opt['performance_prediction']['click_through_rate']:.1%}")
            if opt['recommendations']:
                print(f"    Recommendations: {', '.join(opt['recommendations'])}")
        
        if result["overall_insights"]:
            print(f"\n💡 Insights: {', '.join(result['overall_insights'])}")
    
    # Get final stats
    print(f"\n{len(test_cases) + 2}. Agent Statistics:")
    print("-" * 30)
    stats = client.get_agent_stats()
    print(json.dumps(stats, indent=2))

def run_single_test():
    """Run a single test with custom input"""
    client = AdRewritingAgentClient()
    
    # Example request
    request = {
        "original_text": "Don't miss our flash sale! Limited time only!",
        "target_tone": "professional",
        "target_platforms": ["linkedin", "google", "facebook"],
        "brand_context": "B2B marketing agency",
        "target_audience": "Marketing directors"
    }
    
    print("🚀 Testing Ad Rewriting Agent")
    print("=" * 40)
    print(f"Input: {request['original_text']}")
    print(f"Target Tone: {request['target_tone']}")
    print(f"Platforms: {', '.join(request['target_platforms'])}")
    
    result = client.test_agent(request)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n✅ Rewrite Successful!")
    print(f"Confidence: {result['confidence_score']:.1%}")
    
    print(f"\n📝 Platform Optimizations:")
    for opt in result["platform_optimizations"]:
        print(f"\n{opt['platform'].upper()}:")
        print(f"  {opt['optimized_text']}")
        print(f"  ({opt['character_count']} chars)")
        print(f"  Predicted Performance: {opt['performance_prediction']['click_through_rate']:.1%} CTR")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Single test")
    print("2. Comprehensive test suite")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_single_test()
    elif choice == "2":
        run_comprehensive_test()
    else:
        print("Invalid choice. Running single test by default.")
        run_single_test()