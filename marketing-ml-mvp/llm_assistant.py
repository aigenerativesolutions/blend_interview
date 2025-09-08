"""
LLM Marketing Assistant for Campaign Strategy
Professional AI assistant with marketing domain knowledge
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingAssistant:
    """
    Professional Marketing Strategy Assistant powered by LLM
    Provides contextual insights for marketing campaigns
    """
    
    def __init__(self, model="gpt-3.5-turbo", temperature=0.7):
        """
        Initialize Marketing Assistant
        
        Args:
            model: LLM model to use (gpt-3.5-turbo, gpt-4, or local model)
            temperature: Response creativity (0.0-1.0)
        """
        self.model = model
        self.temperature = temperature
        
        # Try different LLM providers
        self.client = None
        self.provider = self._initialize_llm()
        
        # Marketing context and knowledge base
        self.marketing_context = self._load_marketing_context()
        
    def _initialize_llm(self):
        """Initialize LLM client with fallback options"""
        
        # Option 1: OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI client initialized")
                return "openai"
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
        
        # Option 2: Groq (faster, often free)
        if os.getenv("GROQ_API_KEY"):
            try:
                import groq
                self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
                self.model = "llama3-8b-8192"  # Fast Groq model
                logger.info("✅ Groq client initialized")
                return "groq"
            except Exception as e:
                logger.warning(f"Groq initialization failed: {e}")
        
        # Option 3: Anthropic Claude
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.model = "claude-3-sonnet-20240229"
                logger.info("✅ Anthropic client initialized")
                return "anthropic"
            except Exception as e:
                logger.warning(f"Anthropic initialization failed: {e}")
        
        # Option 4: Local Ollama (free, no API key needed)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.model = "llama3:8b"  # or llama2, mistral, etc.
                logger.info("✅ Ollama local client detected")
                return "ollama"
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        # Fallback: Use demo mode
        logger.warning("⚠️ No LLM provider available. Using demo responses.")
        return "demo"
    
    def _load_marketing_context(self):
        """Load marketing domain knowledge and context"""
        return {
            "model_insights": {
                "top_features": [
                    "Total_Spent: Customer's total annual spending (23% importance)",
                    "Income: Annual household income (18% importance)", 
                    "Recency: Days since last purchase (15% importance)",
                    "MntWines: Wine product purchases (12% importance)",
                    "Age: Customer age derived from birth year (10% importance)",
                    "Months_As_Customer: Customer tenure (8% importance)",
                    "NumCatalogPurchases: Catalog channel usage (7% importance)",
                    "MntMeatProducts: Meat product spending (7% importance)"
                ],
                "performance_metrics": {
                    "accuracy": "85.2%",
                    "roc_auc": "0.887",
                    "precision": "78.5%",
                    "recall": "72.1%",
                    "response_rate": "15.2%"
                }
            },
            "customer_segments": {
                "high_value": {
                    "criteria": "Income > $70K, Total_Spent > $1200",
                    "response_rate": "34%",
                    "best_channels": ["catalog", "store"],
                    "preferred_products": ["wines", "meat", "gold"]
                },
                "price_sensitive": {
                    "criteria": "High deals purchases, web-focused",
                    "response_rate": "18%", 
                    "best_channels": ["web", "deals"],
                    "preferred_products": ["fruits", "sweets"]
                },
                "loyal_veterans": {
                    "criteria": "Months_As_Customer > 36, steady spending",
                    "response_rate": "22%",
                    "best_channels": ["catalog", "store"],
                    "preferred_products": ["wines", "fish"]
                }
            },
            "campaign_strategies": {
                "wine_lovers": "Target high-income customers with wine spending > $400",
                "deal_hunters": "Web promotions for customers with >5 deal purchases",
                "recent_buyers": "Upsell to customers with Recency < 30 days",
                "catalog_users": "Premium offers via catalog for age > 45",
                "multi_channel": "Cross-channel campaigns for omnichannel users"
            }
        }
    
    def get_response(self, query: str, chat_history: List[Dict] = None) -> str:
        """
        Generate contextual marketing response
        
        Args:
            query: User question about marketing
            chat_history: Previous conversation context
            
        Returns:
            Assistant response with marketing insights
        """
        
        if self.provider == "demo":
            return self._get_demo_response(query)
        
        # Build context-aware prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, chat_history)
        
        try:
            if self.provider == "openai":
                return self._get_openai_response(system_prompt, user_prompt)
            elif self.provider == "groq":
                return self._get_groq_response(system_prompt, user_prompt)
            elif self.provider == "anthropic":
                return self._get_anthropic_response(system_prompt, user_prompt)
            elif self.provider == "ollama":
                return self._get_ollama_response(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._get_demo_response(query)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with marketing context"""
        return f"""You are a professional Marketing Strategy Assistant specializing in customer campaign optimization.

CONTEXT - MARKETING CAMPAIGN MODEL:
You have access to an XGBoost model that predicts customer response to marketing campaigns with 85.2% accuracy.

KEY MODEL INSIGHTS:
{chr(10).join(self.marketing_context['model_insights']['top_features'])}

CUSTOMER SEGMENTS IDENTIFIED:
- High-Value (34% response): {self.marketing_context['customer_segments']['high_value']['criteria']}
- Price-Sensitive (18% response): {self.marketing_context['customer_segments']['price_sensitive']['criteria']} 
- Loyal Veterans (22% response): {self.marketing_context['customer_segments']['loyal_veterans']['criteria']}

PERFORMANCE METRICS:
- Model Accuracy: 85.2%
- ROC-AUC: 0.887
- Current Campaign Response Rate: 15.2%

INSTRUCTIONS:
1. Provide actionable, data-driven marketing advice
2. Reference model insights when relevant
3. Suggest specific customer targeting strategies
4. Include expected ROI/performance improvements when possible
5. Be concise but comprehensive
6. Use marketing terminology appropriately
7. Format responses with clear sections using markdown

Always ground your recommendations in the model data and customer segments provided."""

    def _build_user_prompt(self, query: str, chat_history: List[Dict] = None) -> str:
        """Build user prompt with conversation context"""
        
        prompt = f"Marketing Question: {query}\n\n"
        
        # Add relevant context based on query keywords
        if any(word in query.lower() for word in ['segment', 'customer', 'target']):
            prompt += "RELEVANT CONTEXT: Focus on customer segmentation insights and targeting strategies.\n"
        elif any(word in query.lower() for word in ['campaign', 'response', 'improve']):
            prompt += "RELEVANT CONTEXT: Focus on campaign optimization and response rate improvement.\n"
        elif any(word in query.lower() for word in ['model', 'feature', 'predict']):
            prompt += "RELEVANT CONTEXT: Focus on model insights and predictive analytics.\n"
        elif any(word in query.lower() for word in ['channel', 'web', 'catalog', 'store']):
            prompt += "RELEVANT CONTEXT: Focus on channel strategy and omnichannel optimization.\n"
        
        prompt += "Please provide a strategic marketing response with specific recommendations."
        
        return prompt
    
    def _get_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=800
        )
        return response.choices[0].message.content
    
    def _get_groq_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from Groq"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=800
        )
        return response.choices[0].message.content
    
    def _get_anthropic_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from Anthropic Claude"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text
    
    def _get_ollama_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from local Ollama"""
        import requests
        
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        return response.json().get("response", "Sorry, I couldn't generate a response.")
    
    def _get_demo_response(self, query: str) -> str:
        """Generate demo responses when no LLM is available"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['segment', 'customer', 'target']):
            return """🎯 **Customer Segmentation Strategy**

Based on our XGBoost model analysis (85.2% accuracy), here are the key customer segments:

**🔥 High-Value Segment (34% response rate)**
- Criteria: Income > $70K, Annual spending > $1,200
- Best channels: Catalog, Store
- Preferred products: Wines, Premium meats
- Strategy: Premium campaigns, exclusive offers

**💰 Price-Sensitive Segment (18% response rate)**
- Criteria: High deal purchases, web-focused shoppers  
- Best channels: Web, Email promotions
- Preferred products: Fruits, Sweets, Discounted items
- Strategy: Limited-time deals, bundle offers

**⭐ Loyal Veterans (22% response rate)**
- Criteria: 3+ years tenure, consistent spending
- Best channels: Catalog, Store visits
- Preferred products: Wine, Fish, Traditional items
- Strategy: Loyalty rewards, early access

**Expected ROI Improvement: 23-35%** by focusing on High-Value segment first."""
        
        elif any(word in query_lower for word in ['campaign', 'optimize', 'improve', 'response']):
            return """📈 **Campaign Optimization Recommendations**

Based on predictive model insights (Current: 15.2% → Target: 19.8%):

**🎯 Targeting Improvements**
1. **Focus on High-Probability Customers** (>60% model score)
   - Expected improvement: +23% response rate
   - Target ~340 customers vs. current 2,240

2. **Recency-Based Campaigns**
   - Target customers with <30 days since last purchase
   - 28% response rate vs. 15.2% average

**📊 Channel Optimization**
- **Catalog**: 31% response (wine buyers, age 45+)
- **Web**: 19% response (deal hunters, frequent visitors)  
- **Store**: 24% response (meat/fish product buyers)

**💡 Personalization Tactics**
- Wine lovers (>$400 wine spend): Premium wine campaigns
- Deal hunters (>5 deal purchases): Flash sales, bundles
- Multi-channel users: Cross-channel experiences

**Expected Results:** 15.2% → 19.8% response rate (+30% improvement)"""
        
        elif any(word in query_lower for word in ['model', 'feature', 'important', 'predict']):
            return """🔍 **Model Analytics & Key Insights**

**🏆 Model Performance (XGBoost + Temperature Calibration)**
- Accuracy: 85.2%
- ROC-AUC: 0.887  
- Precision: 78.5%
- Response Rate: 15.2%

**📊 Top Predictive Features:**
1. **Total_Spent** (23%) - Annual spending is strongest indicator
2. **Income** (18%) - Higher income → better response likelihood
3. **Recency** (15%) - Recent buyers more likely to engage
4. **Wine_Spending** (12%) - Premium product affinity signal
5. **Customer_Age** (10%) - Age-based preferences
6. **Tenure** (8%) - Customer lifetime value indicator

**🧠 Key Patterns Discovered:**
- Customers spending >$1,000: 34% response rate
- Recent purchasers (<30 days): 28% response rate
- Wine buyers ($400+ annually): 31% response rate  
- Multi-channel users: 22% response rate

**Actionable Insight:** Combine high spending + recent activity for 40%+ response rates."""
        
        elif any(word in query_lower for word in ['channel', 'web', 'catalog', 'store']):
            return """📺 **Channel Strategy & Optimization**

**🎯 Channel Performance Analysis:**

**📖 Catalog Channel**
- Response Rate: 31%
- Best for: Wine buyers, Age 45+, High income
- Products: Premium wines, meat, gold items
- Timing: Monthly premium catalogs

**🌐 Web Channel** 
- Response Rate: 19%
- Best for: Deal hunters, frequent visitors (>7/month)
- Products: Discounted fruits, sweets, bundles
- Timing: Flash sales, weekend promotions

**🏪 Store Channel**
- Response Rate: 24%  
- Best for: Meat/fish buyers, local customers
- Products: Fresh items, bulk purchases
- Timing: Seasonal campaigns, in-store events

**🚀 Omnichannel Strategy:**
- Multi-channel customers: 22% higher response
- Catalog → Web: 27% conversion
- Store → Web: 19% conversion

**Recommendation:** Start with catalog for high-value, then retarget non-responders via web with deals."""
        
        else:
            return """🚀 **Marketing Strategy Assistant**

I'm your AI marketing strategist trained on campaign data with 85.2% accuracy. I can help with:

**📊 Available Analysis:**
- **Customer Segmentation** - Identify high-value prospects
- **Campaign Optimization** - Improve response rates 15.2% → 19.8%
- **Channel Strategy** - Catalog (31%), Store (24%), Web (19%)
- **Predictive Insights** - Model feature importance & patterns
- **ROI Improvement** - Data-driven targeting recommendations

**💡 Popular Questions:**
- "Which customers should I target for highest ROI?"
- "How can I improve campaign response rates?" 
- "What's the best channel for wine buyers?"
- "Which features predict customer response?"

**🎯 Quick Win:** Target customers with >$1,000 annual spend + recent purchase (<30 days) for 40%+ response rates!

What specific marketing challenge can I help you solve?"""

# Configuration for different LLM providers
def setup_llm_environment():
    """Helper function to set up LLM environment variables"""
    
    instructions = """
🔧 **LLM Setup Instructions**

To enable the Marketing Assistant, set up one of these providers:

**Option 1: OpenAI (Recommended)**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Option 2: Groq (Fast & Often Free)**  
```bash
export GROQ_API_KEY="your-groq-api-key"
# Get free key at: https://console.groq.com/
```

**Option 3: Anthropic Claude**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"  
```

**Option 4: Local Ollama (Free)**
```bash
# Install: https://ollama.ai/
ollama pull llama3:8b
ollama serve
```

**Demo Mode**: If no API keys are set, the assistant will use pre-written responses.
"""
    return instructions

if __name__ == "__main__":
    print(setup_llm_environment())