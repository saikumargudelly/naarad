"""Creativity Agent for generating creative content, brainstorming, and innovative solutions."""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import random
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import os

from .base import BaseAgent, AgentConfig
from agent.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class CreativityType(Enum):
    """Types of creative tasks."""
    BRAINSTORMING = "brainstorming"
    STORYTELLING = "storytelling"
    PROBLEM_SOLVING = "problem_solving"
    ARTISTIC_INSPIRATION = "artistic_inspiration"
    INNOVATION = "innovation"
    DESIGN_THINKING = "design_thinking"
    CREATIVE_WRITING = "creative_writing"
    IDEA_GENERATION = "idea_generation"

@dataclass
class CreativeIdea:
    """Represents a creative idea with metadata."""
    title: str
    description: str
    category: str
    novelty_score: float  # 0.0 to 1.0
    feasibility_score: float  # 0.0 to 1.0
    impact_score: float  # 0.0 to 1.0
    tags: List[str]
    implementation_steps: List[str]

class CreativityAgent(BaseAgent):
    """Agent specialized in creative thinking and idea generation.
    Modular, stateless, and uses injected memory manager for context/state.
    """
    def __init__(self, config: AgentConfig, memory_manager: MemoryManager = None):
        super().__init__(config)
        self.memory_manager = memory_manager
        logger.info(f"CreativityAgent initialized with memory_manager: {bool(memory_manager)}")
        self.creativity_techniques = self._initialize_creativity_techniques()
        self.idea_templates = self._load_idea_templates()
        self.creative_history = []
        
    def _initialize_creativity_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize various creativity techniques and frameworks."""
        return {
            "scamper": {
                "name": "SCAMPER Technique",
                "description": "Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse",
                "questions": [
                    "What can be substituted?",
                    "What can be combined?",
                    "What can be adapted?",
                    "What can be modified?",
                    "What can be put to another use?",
                    "What can be eliminated?",
                    "What can be reversed?"
                ]
            },
            "six_thinking_hats": {
                "name": "Six Thinking Hats",
                "description": "Different perspectives: White (facts), Red (emotions), Black (caution), Yellow (benefits), Green (creativity), Blue (process)",
                "hats": [
                    {"color": "white", "focus": "facts and information"},
                    {"color": "red", "focus": "emotions and feelings"},
                    {"color": "black", "focus": "caution and potential problems"},
                    {"color": "yellow", "focus": "benefits and positive aspects"},
                    {"color": "green", "focus": "creativity and new ideas"},
                    {"color": "blue", "focus": "process and organization"}
                ]
            },
            "mind_mapping": {
                "name": "Mind Mapping",
                "description": "Visual brainstorming technique connecting related ideas",
                "steps": [
                    "Start with central concept",
                    "Branch out with related ideas",
                    "Connect and expand branches",
                    "Add visual elements",
                    "Explore connections"
                ]
            },
            "random_stimulation": {
                "name": "Random Stimulation",
                "description": "Use random words or concepts to spark new ideas",
                "stimuli": [
                    "nature", "technology", "art", "music", "food", "travel",
                    "sports", "science", "history", "fantasy", "space", "ocean",
                    "mountains", "cities", "animals", "colors", "shapes", "textures"
                ]
            },
            "reverse_thinking": {
                "name": "Reverse Thinking",
                "description": "Consider the opposite or reverse of the problem",
                "questions": [
                    "What if we did the opposite?",
                    "What would make this fail?",
                    "What's the worst possible outcome?",
                    "How can we make this worse?",
                    "What assumptions can we challenge?"
                ]
            }
        }
    
    def _load_idea_templates(self) -> Dict[str, List[str]]:
        """Load templates for different types of creative ideas."""
        return {
            "product_idea": [
                "A {category} that {benefit} by {mechanism}",
                "An innovative {category} combining {feature1} and {feature2}",
                "A smart {category} that adapts to {context}",
                "A sustainable {category} using {material/technology}"
            ],
            "service_idea": [
                "A {service_type} that {value_proposition}",
                "Personalized {service_type} based on {personalization_factor}",
                "On-demand {service_type} for {target_audience}",
                "Community-driven {service_type} platform"
            ],
            "story_idea": [
                "A {protagonist} who {conflict} in {setting}",
                "When {inciting_incident}, {character} must {goal}",
                "In a world where {premise}, {character} discovers {revelation}",
                "A {genre} story about {theme} set in {time_period}"
            ],
            "solution_idea": [
                "Solve {problem} by {approach}",
                "Address {challenge} through {methodology}",
                "Overcome {obstacle} using {strategy}",
                "Transform {current_state} into {desired_state} via {process}"
            ]
        }
    
    async def process(self, input_text: str, context: Dict[str, Any] = None, conversation_id: str = None, user_id: str = None, conversation_memory=None, **kwargs) -> Dict[str, Any]:
        logger.info(f"CreativityAgent.process called | input_text: {input_text} | conversation_id: {conversation_id} | user_id: {user_id}")
        try:
            chat_history = kwargs.get('chat_history', '')
            topic = None
            intent = None
            last_user_message = None
            if conversation_memory:
                topic = conversation_memory.topics[-1] if conversation_memory.topics else None
                intent = conversation_memory.intents[-1] if conversation_memory.intents else None
                for msg in reversed(conversation_memory.messages):
                    if msg['role'] == 'user':
                        last_user_message = msg['content']
                        break
            # Compose a context-aware prompt
            context_snippets = "\n".join([
                f"{m['role'].capitalize()}: {m['content']}" for m in conversation_memory.messages[-6:]
            ]) if conversation_memory else ""
            system_prompt = (
                "You are a creative assistant. Use the conversation context, topic, and intent to answer the user's question as creatively and helpfully as possible. "
                "If the user is following up, use the previous context to disambiguate."
            )
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            from llm.config import settings
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Conversation context:\n{context_snippets}\n\nTopic: {topic}\nIntent: {intent}\n\nUser question: {input_text}")
            ]
            llm = ChatGroq(
                temperature=0.2,
                model_name=settings.REASONING_MODEL,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
            result = await llm.ainvoke(messages)
            return {"output": result.content.strip(), "metadata": {"success": True, "topic": topic, "intent": intent}}
        except Exception as e:
            logger.error(f"Async error in creativity process: {str(e)}", exc_info=True)
            return {
                'output': f"I encountered an error while processing your creativity request: {str(e)}",
                'metadata': {
                    'error': str(e),
                    'success': False
                }
            }
    
    def _classify_creativity_request(self, text: str) -> CreativityType:
        """Classify the type of creativity request."""
        text_lower = text.lower()
        
        # Brainstorming keywords
        if any(word in text_lower for word in ['brainstorm', 'ideas', 'suggestions', 'options', 'alternatives']):
            return CreativityType.BRAINSTORMING
        
        # Storytelling keywords
        if any(word in text_lower for word in ['story', 'narrative', 'plot', 'character', 'fiction', 'tale']):
            return CreativityType.STORYTELLING
        
        # Problem solving keywords
        if any(word in text_lower for word in ['solve', 'problem', 'issue', 'challenge', 'fix', 'resolve']):
            return CreativityType.PROBLEM_SOLVING
        
        # Artistic inspiration keywords
        if any(word in text_lower for word in ['art', 'design', 'creative', 'inspiration', 'visual', 'aesthetic']):
            return CreativityType.ARTISTIC_INSPIRATION
        
        # Innovation keywords
        if any(word in text_lower for word in ['innovate', 'new', 'breakthrough', 'revolutionary', 'disruptive']):
            return CreativityType.INNOVATION
        
        # Default to general creativity
        return CreativityType.IDEA_GENERATION
    
    async def _generate_brainstorming_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate brainstorming ideas using various techniques."""
        ideas = []
        
        # Use SCAMPER technique
        scamper_ideas = self._apply_scamper_technique(text)
        ideas.extend(scamper_ideas)
        
        # Use random stimulation
        random_ideas = self._apply_random_stimulation(text)
        ideas.extend(random_ideas)
        
        # Use reverse thinking
        reverse_ideas = self._apply_reverse_thinking(text)
        ideas.extend(reverse_ideas)
        
        # Format response
        response = "🎯 **Brainstorming Ideas**\n\n"
        response += f"Based on your request: *{text}*\n\n"
        
        for i, idea in enumerate(ideas[:10], 1):  # Limit to top 10 ideas
            response += f"**{i}. {idea['title']}**\n"
            response += f"   {idea['description']}\n"
            response += f"   💡 Novelty: {idea['novelty_score']:.1f} | Feasibility: {idea['feasibility_score']:.1f} | Impact: {idea['impact_score']:.1f}\n\n"
        
        response += "**Next Steps:**\n"
        response += "• Choose 2-3 ideas to explore further\n"
        response += "• Consider combining elements from different ideas\n"
        response += "• Think about implementation challenges and solutions\n"
        
        return response
    
    def _apply_scamper_technique(self, text: str) -> List[CreativeIdea]:
        """Apply SCAMPER technique to generate ideas."""
        ideas = []
        scamper = self.creativity_techniques["scamper"]
        
        for question in scamper["questions"]:
            # Generate idea based on SCAMPER question
            idea = CreativeIdea(
                title=f"SCAMPER: {question.split('?')[0]}",
                description=f"Consider how to {question.lower()} for: {text}",
                category="brainstorming",
                novelty_score=random.uniform(0.6, 0.9),
                feasibility_score=random.uniform(0.5, 0.8),
                impact_score=random.uniform(0.4, 0.7),
                tags=["scamper", "brainstorming"],
                implementation_steps=[
                    "Identify current elements",
                    "Apply SCAMPER question",
                    "Generate variations",
                    "Evaluate potential"
                ]
            )
            ideas.append(idea)
        
        return ideas
    
    def _apply_random_stimulation(self, text: str) -> List[CreativeIdea]:
        """Apply random stimulation technique."""
        ideas = []
        stimuli = self.creativity_techniques["random_stimulation"]["stimuli"]
        
        for _ in range(3):  # Generate 3 random stimulation ideas
            stimulus = random.choice(stimuli)
            idea = CreativeIdea(
                title=f"Random Stimulation: {stimulus.title()}",
                description=f"Use '{stimulus}' as inspiration for: {text}",
                category="brainstorming",
                novelty_score=random.uniform(0.7, 0.95),
                feasibility_score=random.uniform(0.3, 0.7),
                impact_score=random.uniform(0.5, 0.8),
                tags=["random_stimulation", stimulus],
                implementation_steps=[
                    f"Research {stimulus} concepts",
                    "Find connections to your topic",
                    "Generate analogies",
                    "Apply insights"
                ]
            )
            ideas.append(idea)
        
        return ideas
    
    def _apply_reverse_thinking(self, text: str) -> List[CreativeIdea]:
        """Apply reverse thinking technique."""
        ideas = []
        questions = self.creativity_techniques["reverse_thinking"]["questions"]
        
        for question in questions[:3]:  # Use first 3 questions
            idea = CreativeIdea(
                title=f"Reverse Thinking: {question.split('?')[0]}",
                description=f"Consider: {question} for {text}",
                category="problem_solving",
                novelty_score=random.uniform(0.8, 0.95),
                feasibility_score=random.uniform(0.4, 0.6),
                impact_score=random.uniform(0.6, 0.9),
                tags=["reverse_thinking", "challenge_assumptions"],
                implementation_steps=[
                    "Identify current approach",
                    "Consider the opposite",
                    "Analyze failure points",
                    "Generate counter-solutions"
                ]
            )
            ideas.append(idea)
        
        return ideas
    
    async def _generate_story_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate creative story ideas."""
        templates = self.idea_templates["story_idea"]
        ideas = []
        
        # Generate story ideas using templates
        for i, template in enumerate(templates):
            # Fill template with creative elements
            story_idea = template.format(
                protagonist=random.choice(["detective", "scientist", "artist", "explorer", "student", "chef"]),
                conflict=random.choice(["discovers a hidden truth", "faces an impossible choice", "must save someone", "challenges authority"]),
                setting=random.choice(["futuristic city", "small town", "space station", "magical forest", "underwater world"]),
                inciting_incident=random.choice(["a mysterious letter arrives", "technology fails", "a stranger appears", "an ancient artifact is found"]),
                character=random.choice(["hero", "protagonist", "main character", "central figure"]),
                goal=random.choice(["find the truth", "save the world", "prove innocence", "find love", "achieve redemption"]),
                premise=random.choice(["magic is real", "time travel exists", "dreams are connected", "animals can talk"]),
                revelation=random.choice(["their true identity", "a hidden power", "a family secret", "the meaning of life"]),
                genre=random.choice(["mystery", "fantasy", "sci-fi", "romance", "thriller", "adventure"]),
                theme=random.choice(["love", "redemption", "identity", "justice", "freedom", "sacrifice"]),
                time_period=random.choice(["medieval times", "the 1920s", "the future", "present day", "ancient times"])
            )
            
            ideas.append({
                "title": f"Story Idea {i+1}",
                "description": story_idea,
                "novelty_score": random.uniform(0.6, 0.9),
                "feasibility_score": random.uniform(0.7, 0.9),
                "impact_score": random.uniform(0.5, 0.8)
            })
        
        response = "📚 **Creative Story Ideas**\n\n"
        response += f"Inspired by: *{text}*\n\n"
        
        for i, idea in enumerate(ideas, 1):
            response += f"**{i}. {idea['title']}**\n"
            response += f"   {idea['description']}\n"
            response += f"   📖 Creativity: {idea['novelty_score']:.1f} | Feasibility: {idea['feasibility_score']:.1f} | Impact: {idea['impact_score']:.1f}\n\n"
        
        return response
    
    async def _generate_solution_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate innovative problem-solving ideas."""
        templates = self.idea_templates["solution_idea"]
        ideas = []
        
        # Generate solution ideas using templates
        for i, template in enumerate(templates):
            solution_idea = template.format(
                problem=text,
                approach=random.choice(["innovative technology", "community collaboration", "systematic analysis", "creative thinking"]),
                challenge=text,
                methodology=random.choice(["design thinking", "agile development", "scientific method", "creative problem solving"]),
                obstacle=text,
                strategy=random.choice(["break it down", "find alternatives", "leverage resources", "think outside the box"]),
                current_state=text,
                desired_state="improved situation",
                process=random.choice(["iterative improvement", "radical innovation", "incremental change", "paradigm shift"])
            )
            
            ideas.append({
                "title": f"Solution {i+1}",
                "description": solution_idea,
                "novelty_score": random.uniform(0.7, 0.95),
                "feasibility_score": random.uniform(0.6, 0.9),
                "impact_score": random.uniform(0.8, 0.95)
            })
        
        response = "🔧 **Innovative Solutions**\n\n"
        response += f"Problem: *{text}*\n\n"
        
        for i, idea in enumerate(ideas, 1):
            response += f"**{i}. {idea['title']}**\n"
            response += f"   {idea['description']}\n"
            response += f"   ⚡ Innovation: {idea['novelty_score']:.1f} | Feasibility: {idea['feasibility_score']:.1f} | Impact: {idea['impact_score']:.1f}\n\n"
        
        response += "**Implementation Framework:**\n"
        response += "1. Define success criteria\n"
        response += "2. Create action plan\n"
        response += "3. Test and iterate\n"
        response += "4. Measure results\n"
        
        return response
    
    async def _generate_artistic_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate artistic inspiration ideas."""
        response = "🎨 **Artistic Inspiration**\n\n"
        response += f"Inspired by: *{text}*\n\n"
        
        artistic_concepts = [
            "**Color Palette:** Create a mood board with colors that represent the emotions and themes",
            "**Texture Exploration:** Experiment with different materials and surfaces",
            "**Composition Studies:** Explore various layouts and arrangements",
            "**Style Fusion:** Combine different artistic styles and techniques",
            "**Symbolic Elements:** Incorporate meaningful symbols and metaphors",
            "**Light and Shadow:** Play with lighting effects and contrast",
            "**Movement and Flow:** Create dynamic, flowing compositions",
            "**Minimalist Approach:** Strip down to essential elements"
        ]
        
        for i, concept in enumerate(artistic_concepts, 1):
            response += f"{i}. {concept}\n"
        
        response += "\n**Creative Process:**\n"
        response += "• Research and gather inspiration\n"
        response += "• Sketch and experiment\n"
        response += "• Refine and develop\n"
        response += "• Finalize and present\n"
        
        return response
    
    async def _generate_innovation_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate breakthrough innovation ideas."""
        response = "🚀 **Breakthrough Innovations**\n\n"
        response += f"Focus area: *{text}*\n\n"
        
        innovation_areas = [
            "**Disruptive Technology:** Challenge existing paradigms with new approaches",
            "**Cross-Industry Application:** Apply solutions from other fields",
            "**User-Centric Design:** Focus on unmet user needs and pain points",
            "**Sustainability Integration:** Incorporate environmental considerations",
            "**AI and Automation:** Leverage artificial intelligence for new capabilities",
            "**Collaborative Ecosystems:** Build partnerships and networks",
            "**Data-Driven Insights:** Use analytics to inform innovation",
            "**Rapid Prototyping:** Fast iteration and testing cycles"
        ]
        
        for i, area in enumerate(innovation_areas, 1):
            response += f"{i}. {area}\n"
        
        response += "\n**Innovation Framework:**\n"
        response += "1. Identify opportunity spaces\n"
        response += "2. Generate radical ideas\n"
        response += "3. Prototype and test\n"
        response += "4. Scale and implement\n"
        
        return response
    
    async def _generate_general_creative_ideas(self, text: str, context: Dict[str, Any]) -> str:
        """Generate general creative ideas."""
        response = "💡 **Creative Ideas**\n\n"
        response += f"Request: *{text}*\n\n"
        
        # Use multiple creativity techniques
        brainstorming_ideas = await self._generate_brainstorming_ideas(text, context)
        story_ideas = await self._generate_story_ideas(text, context)
        
        response += "**Quick Brainstorming:**\n"
        response += brainstorming_ideas.split("**Next Steps:**")[0] + "\n"
        
        response += "**Creative Story Elements:**\n"
        response += story_ideas.split("**Implementation Framework:**")[0] + "\n"
        
        return response
    
    def _get_used_techniques(self, creativity_type: CreativityType) -> List[str]:
        """Get the creativity techniques used for this type."""
        technique_mapping = {
            CreativityType.BRAINSTORMING: ["scamper", "random_stimulation", "reverse_thinking"],
            CreativityType.STORYTELLING: ["mind_mapping", "random_stimulation"],
            CreativityType.PROBLEM_SOLVING: ["six_thinking_hats", "reverse_thinking"],
            CreativityType.ARTISTIC_INSPIRATION: ["random_stimulation", "mind_mapping"],
            CreativityType.INNOVATION: ["reverse_thinking", "random_stimulation"],
            CreativityType.IDEA_GENERATION: ["scamper", "random_stimulation", "mind_mapping"]
        }
        
        return technique_mapping.get(creativity_type, ["general_creativity"])
    
    def _update_creative_history(self, creativity_type: CreativityType, input_text: str, response: str):
        """Update creative history for pattern analysis."""
        self.creative_history.append({
            'type': creativity_type.value,
            'input': input_text,
            'response_length': len(response),
            'timestamp': datetime.utcnow().isoformat(),
            'techniques_used': self._get_used_techniques(creativity_type)
        })
        
        # Keep only recent history (last 50 interactions)
        if len(self.creative_history) > 50:
            self.creative_history = self.creative_history[-50:] 