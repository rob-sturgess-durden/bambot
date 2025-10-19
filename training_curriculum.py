"""
Structured training curriculum for evangelism training.
Each topic is broken into discrete steps with clear learning objectives.
"""

TRAINING_CURRICULUM = {
    "1": {
        "title": "Theology behind effective outreach",
        "steps": [
            {
                "step": 1,
                "objective": "Explore pressure in evangelism",
                "prompt": "Ask the learner if they've ever felt pressure to convince someone about their faith. Listen to their response and validate their experience.",
                "completion_signal": "User shares their experience with feeling pressure or not"
            },
            {
                "step": 2,
                "objective": "Introduce John 6:44 principle",
                "prompt": "Introduce John 6:44: 'No one can come to me unless the Father who sent me draws them.' Ask what they think this means for our role in evangelism.",
                "completion_signal": "User responds with their interpretation"
            },
            {
                "step": 3,
                "objective": "Clarify our role vs God's role",
                "prompt": "Explain: Your job isn't to convince anyone - it's to explore where they are spiritually and find people the Father is already drawing. Ask if this changes how they view evangelism.",
                "completion_signal": "User acknowledges the shift in perspective"
            },
            {
                "step": 4,
                "objective": "Introduce Woman at the Well case study",
                "prompt": "Briefly introduce the Woman at the Well story (John 4). Ask them what they notice about how Jesus approached her - did He pressure or explore?",
                "completion_signal": "User engages with the story"
            },
            {
                "step": 5,
                "objective": "Apply the principle",
                "prompt": "Ask: How might this principle change your next conversation with someone far from God? Listen and encourage their thoughts.",
                "completion_signal": "User shares practical application ideas"
            },
            {
                "step": 6,
                "objective": "Summary and completion",
                "prompt": "Summarize key points: God draws people, our job is to explore and share our story. Ask if they have any final questions. If ready, offer to mark topic complete.",
                "completion_signal": "User confirms understanding and readiness to complete"
            }
        ]
    },
    "2": {
        "title": "Key principles about people far from God",
        "steps": [
            {
                "step": 1,
                "objective": "Introduce traffic light concept",
                "prompt": "Ask: Have you noticed that people respond differently when you talk about faith? Some are open, some curious, some resistant?",
                "completion_signal": "User acknowledges different responses"
            },
            {
                "step": 2,
                "objective": "Explain Red Light",
                "prompt": "Introduce Red Light: Someone not interested right now (through you). Explain they might be interested later or through someone else. Ask what they think happens if we push too hard with Red Lights.",
                "completion_signal": "User understands Red Light concept"
            },
            {
                "step": 3,
                "objective": "How to respond to Red Lights",
                "prompt": "Teach: With Red Lights, be loving and respectful (1 Cor 13, 1 Peter 3). This keeps them open for future conversations. Ask them to describe what a Red Light looks like and how they'd respond.",
                "completion_signal": "User can describe Red Light response"
            },
            {
                "step": 4,
                "objective": "Explain Green Light",
                "prompt": "Introduce Green Light: Someone ready to decide right now! They exist - maybe 1 in 20. The key: they don't know how to ask what to do next. You need to ask them. What do you think we should ask?",
                "completion_signal": "User suggests or hears the decision question"
            },
            {
                "step": 5,
                "objective": "Green Light response skill",
                "prompt": "Teach the question: 'Would you like to pray to receive Christ right now?' or 'Would you like to fix your relationship with God right now?' Ask them to describe a Green Light scenario.",
                "completion_signal": "User can describe Green Light and response"
            },
            {
                "step": 6,
                "objective": "Explain Yellow Light",
                "prompt": "Introduce Yellow Light: Spiritually curious but unsure about next steps or trusting you. They want to explore safely. Ask what they think we should offer Yellow Lights.",
                "completion_signal": "User suggests or hears about offering to learn more"
            },
            {
                "step": 7,
                "objective": "Yellow Light invitation skill",
                "prompt": "Teach: Offer an invitation to learn more - a discussion group, Bible study, or meeting. Ask them to think of what they could invite someone to.",
                "completion_signal": "User identifies invitation options"
            },
            {
                "step": 8,
                "objective": "Unchurched vs Dechurched",
                "prompt": "Explain: Yellow/Green Lights fall into Unchurched (never been to church) or Dechurched (familiar with church). This affects what you invite them to. Ask if they'd like to know why.",
                "completion_signal": "User shows interest or understanding"
            },
            {
                "step": 9,
                "objective": "Summary and completion",
                "prompt": "Summarize: Red (not now, be kind), Yellow (curious, invite to learn), Green (ready, ask decision question). Unchurched may need different approach than Dechurched. Any questions?",
                "completion_signal": "User confirms understanding"
            }
        ]
    },
    "3": {
        "title": "Go-to skills for conversations",
        "steps": [
            {
                "step": 1,
                "objective": "Introduce the skills list",
                "prompt": "We'll practice 5 key skills: Transition questions, 3-minute testimony, 1-minute testimony, answered prayer story, and invitation skills. Ready to start?",
                "completion_signal": "User is ready to begin"
            },
            {
                "step": 2,
                "objective": "Transition/Segue skill",
                "prompt": "Transition skill: Questions that naturally lead to spiritual topics. Ask: Give me a list of questions you don't mind strangers asking you (like about hobbies, weekend plans, etc).",
                "completion_signal": "User provides list of questions"
            },
            {
                "step": 3,
                "objective": "Practice transition questions",
                "prompt": "Good! These help you learn about someone before sharing. Now brainstorm: How could you transition from 'What do you do for fun?' to mentioning church or faith naturally?",
                "completion_signal": "User suggests transition ideas"
            },
            {
                "step": 4,
                "objective": "3-Minute Testimony skill",
                "prompt": "3-Minute Testimony: Your faith story. Tell me - How did you get serious about Jesus? What was your life like before that moment? (Don't worry about time, just share.)",
                "completion_signal": "User shares their testimony"
            },
            {
                "step": 5,
                "objective": "Feedback on testimony",
                "prompt": "Thank you for sharing! That's powerful. The key is: Before/How/After structure. Keep it to 3 minutes max. Practice will make this natural. Ready for the next skill?",
                "completion_signal": "User ready to continue"
            },
            {
                "step": 6,
                "objective": "1-Minute Testimony skill",
                "prompt": "1-Minute version: Sometimes you only have a brief moment. In 5 sentences or less, what's the most important part of your faith journey?",
                "completion_signal": "User shares condensed testimony"
            },
            {
                "step": 7,
                "objective": "Answered Prayer Story skill",
                "prompt": "Answered Prayer Story: A specific time God answered prayer in a surprising way. This shows God is real and active. Can you think of one from your life?",
                "completion_signal": "User shares answered prayer story"
            },
            {
                "step": 8,
                "objective": "Invitation Skills",
                "prompt": "Invitation Skills: What could you invite people to if they wanted to learn more about Jesus? Think about what's available to you (small group, church service, Bible study, coffee chat).",
                "completion_signal": "User lists invitation options"
            },
            {
                "step": 9,
                "objective": "Summary and completion",
                "prompt": "Great work! You now have 5 tools: Transition questions, 3-min testimony, 1-min testimony, answered prayer story, and invitations. Practice these! Any final questions?",
                "completion_signal": "User confirms understanding"
            }
        ]
    },
    "4": {
        "title": "Dos and don'ts",
        "steps": [
            {
                "step": 1,
                "objective": "Introduce hot tips",
                "prompt": "Let's cover some practical tips to make your conversations smoother. Ready?",
                "completion_signal": "User is ready"
            },
            {
                "step": 2,
                "objective": "Ask questions early tip",
                "prompt": "Tip 1: Ask lots of questions about the other person early. Why? Because after they talk about themselves, it's natural for you to talk about yourself - including your values and faith. Does this make sense? Have you experienced this?",
                "completion_signal": "User responds to the tip"
            },
            {
                "step": 3,
                "objective": "Avoid Christian jargon tip",
                "prompt": "Tip 2: Avoid 'Christian talk.' Instead of 'propitiating sacrifice,' say 'I needed help to become the person God wants me to be.' Talk about faith like your favorite food or hobby. Thoughts?",
                "completion_signal": "User understands and responds"
            },
            {
                "step": 4,
                "objective": "Keep it short tip",
                "prompt": "Tip 3: Keep comments short. No sermons! Stories under 3 minutes. Ask permission for longer stories and make them worth it. Ever been on the receiving end of a long sermon?",
                "completion_signal": "User responds to tip"
            },
            {
                "step": 5,
                "objective": "Summary and completion",
                "prompt": "To recap: Ask questions first, avoid jargon, keep it short. These make conversations natural and inviting. Any questions on the dos and don'ts?",
                "completion_signal": "User confirms understanding"
            }
        ]
    }
}


def get_topic_info(topic_id: str) -> dict:
    """Get topic information by ID"""
    return TRAINING_CURRICULUM.get(topic_id, {})


def get_step_info(topic_id: str, step_number: int) -> dict:
    """Get specific step information"""
    topic = get_topic_info(topic_id)
    if not topic or "steps" not in topic:
        return {}

    steps = topic["steps"]
    for step in steps:
        if step["step"] == step_number:
            return step
    return {}


def get_total_steps(topic_id: str) -> int:
    """Get total number of steps for a topic"""
    topic = get_topic_info(topic_id)
    if not topic or "steps" not in topic:
        return 0
    return len(topic.get("steps", []))


def is_topic_complete(topic_id: str, current_step: int) -> bool:
    """Check if user has completed all steps for a topic"""
    total_steps = get_total_steps(topic_id)
    return current_step >= total_steps


def build_training_context(topic_id: str, current_step: int, user_response: str = None) -> str:
    """Build context for the AI trainer based on current progress"""
    topic = get_topic_info(topic_id)
    if not topic:
        return "Invalid topic selected."

    step = get_step_info(topic_id, current_step)
    if not step:
        return f"Topic '{topic['title']}' completed or invalid step."

    total_steps = get_total_steps(topic_id)

    context_parts = [
        f"=== TRAINING SESSION ===",
        f"Topic: {topic['title']}",
        f"Step {current_step} of {total_steps}",
        f"",
        f"OBJECTIVE FOR THIS STEP:",
        f"{step['objective']}",
        f"",
        f"INSTRUCTIONS FOR THIS STEP:",
        f"{step['prompt']}",
        f"",
        f"COMPLETION SIGNAL:",
        f"Move to next step when: {step['completion_signal']}",
        f"",
        f"IMPORTANT:",
        f"- Stay focused on THIS step only",
        f"- Don't jump ahead to future content",
        f"- After the user responds appropriately, acknowledge briefly and then stop responding",
        f"- Do NOT ask another question or continue teaching - wait for the system to advance to the next step",
    ]

    if current_step == total_steps:
        context_parts.extend([
            f"",
            f"*** THIS IS THE FINAL STEP ***",
            f"After completing this step, offer to mark the topic as complete.",
        ])

    if user_response:
        context_parts.extend([
            f"",
            f"USER'S RESPONSE:",
            f"{user_response}"
        ])

    return "\n".join(context_parts)
