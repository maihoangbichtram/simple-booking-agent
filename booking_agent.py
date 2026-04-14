from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from typing import TypedDict, Annotated, List, Optional

from utils import generate_time_slots

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",  # or your deployment
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # streaming=True,
    # tags=['graph']
    # other params...
)

class BookingState(TypedDict):
    """State for the booking conversation."""
    messages: List[dict]
    stage: str  # greeting, select_speciality, select_doctor, select_slot, confirm, completed
    selected_date: Optional[str]
    selected_slot: Optional[str]
    customer_name: Optional[str]
    customer_phone: Optional[str]
    available_options: List[str]  # For clickable UI options

def create_initial_state():
    """Create initial state for the conversation."""
    return {
        "messages": [],
        "stage": "greeting",
        "selected_date": None,
        "selected_slot": None,
        "customer_name": None,
        "customer_phone": None,
        "booking_id": None,
        "available_options": []
    }

# Define valid routes for each stage (used to prevent invalid routing)
VALID_ROUTES_PER_STAGE = {
    "greeting": {"greeting", "select_speciality", "cancelled"},
    "select_date": {"select_date", "select_slot"},
    "select_slot": {"select_slot", "confirm"},
    "confirm": {"confirm", "collect_details", "cancelled", "select_slot"},
    "collect_details": {"collect_details", "completed"}
}

def call_llm(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 50
):
    """
    Centralized helper for all LLM calls.
    Returns the assistant's response
    """
    try:
        system_prompt = SystemMessage(
            system_prompt
        )

        response = llm.bind(max_tokens=max_tokens).invoke(
            [system_prompt] + [user_prompt]
        )

        print(f"LLM response: {response}")
        return response
    except Exception as e:
        print(f"LLM call error: {e}")
        return ""


def llm_router(state: BookingState, k=4) -> str:
    """
    Routes the conversation based on user intent while enforcing guardrails.

    Logic:
    1. Check if message is on-topic; if not, add guardrail response and stay in current stage
    2. Route based on stage-specific logic
    3. Validate route is allowed for current stage; if not, default to current stage
    """
    current_stage = state.get("stage", "greeting")
    messages = state.get("messages", [])
    recent_messages = messages[-k:]
    conversation_snippet = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in recent_messages)

    # ===== ROUTING LOGIC: Route based on current stage and user message =====
    routing_prompts = {
        "confirm": f"""Analyze the user's response: "{conversation_snippet}"
                    Does the user want to proceed with this appointment? 
                    Respond with ONLY: 
                    - 'collect_details' if they say yes, okay, confirm, or agree.
                    - 'cancelled' if they say no, cancel, or stop.
                    - 'select_slot' if they want to change the time or pick a different slot.
                    - 'confirm' if it is unclear and you need to ask again.
                    """
    }

    # If no routing prompt for this stage, return current stage (node handles transitions)
    if current_stage not in routing_prompts:
        return current_stage

    try:
        response = call_llm(
            system_prompt="You are a conversational AI routing expert. Always respond with ONLY the exact route name.",
            user_prompt=routing_prompts[current_stage],
            max_tokens=20
        )
        route = response.content.strip().lower().replace("'", "").replace('"', "").strip()
        print(f"Routing decision: {route}")
    except Exception as e:
        print(f"⚠️ Routing error: {e}. Defaulting to {current_stage}")
        return current_stage

    # ===== VALIDATION: Ensure route is valid for current stage =====
    valid_routes = VALID_ROUTES_PER_STAGE.get(current_stage, {current_stage})

    if route not in valid_routes:
        print(
            f"⚠️ Invalid route '{route}' for stage '{current_stage}'. Valid: {valid_routes}. Defaulting to '{current_stage}'")
        return current_stage

    return route


def greeting_node(state: BookingState) -> BookingState:
    """Greets the user and pauses to see if they want to book."""
    state["stage"] = "greeting"

    # Default message
    msg = "👋 Welcome! Would you like to book an appointment?"

    # If this is the first time in this node (no messages yet or last message was user)
    if not state["messages"] or state["messages"][-1]["role"] != "assistant":
        state["messages"].append({
            "role": "assistant",
            "content": msg,
            "options": ["Book Appointment"]
        })

    user_input = interrupt({
        "role": "assistant",
        "content": msg,
        "available_options": ["Book Appointment"]
    })
    state["messages"].append({"role": "user", "content": user_input})
    return state

def select_date_node(state: BookingState) -> BookingState:
    """Handle date selection using interrupt and LLM extraction."""
    state["stage"] = "select_date"

    # Simple date options for demo
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now().replace(day=datetime.now().day + 1)).strftime("%Y-%m-%d")
    available_dates = ["Today", "Tomorrow"]

    title = f"When would you like to visit? We have slots for Today ({today}) and Tomorrow ({tomorrow})."
    if not state["messages"] or state["messages"][-1]["content"] != title:
        state["messages"].append({
            "role": "assistant",
            "content": title,
            "options": available_dates
        })
    raw_user_input = interrupt({
        "role": "assistant",
        "content": title,
        "available_options": available_dates
    })

    prompt = f"""Extract the date from: "{raw_user_input}"
    Relative references: Today is {today}, Tomorrow is {tomorrow}.
    Respond with ONLY the date in YYYY-MM-DD format or "UNKNOWN"."""

    try:
        response = call_llm(
            system_prompt="You extract dates from messages.",
            user_prompt=prompt,
            max_tokens=20
        )
        extracted = response.content.strip()

        # Simple mapping for common inputs if LLM output is slightly off
        selected = extracted if extracted != "UNKNOWN" else None

        if selected:
            state["selected_date"] = selected
            state["messages"].append({"role": "user", "content": raw_user_input})
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "I'm sorry, I couldn't understand the date. Could you please specify if you want Today or Tomorrow?"
            })
    except Exception as e:
        print(f"Date extraction error: {e}")

    return state

def select_slot_node(state: BookingState) -> BookingState:
    """Handle time slot selection using interrupt and LLM extraction."""
    state["stage"] = "select_slot"
    available_slots = generate_time_slots()

    message = f"Pick a slot: {', '.join(available_slots)}"

    if state["messages"][-1]["content"] != message:
        state["messages"].append({
            "role": "assistant",
            "content": message,
            "options": available_slots
        })

    raw_user_input = interrupt({
        "role": "assistant",
        "content": message,
        "available_options": available_slots
    })

    prompt = f"""Extract the time slot from: "{raw_user_input}"
    Available: {', '.join(available_slots)}
    Respond with ONLY the exact time slot or "UNKNOWN"."""

    try:
        response = call_llm(
            system_prompt="You extract time slots from messages.",
            user_prompt=prompt,
            max_tokens=20
        )
        extracted = response.content.strip()

        selected = next((s for s in available_slots if s.lower() in extracted.lower()), None)

        if selected:
            state["selected_slot"] = selected
            state["messages"].append({"role": "user", "content": raw_user_input})
        else:
            state["messages"].append({
                "role": "assistant",
                "content": "I didn't catch that. Which slot works?"
            })
    except Exception as e:
        print(f"Slot extraction error: {e}")

    return state

def confirm_node(state: BookingState) -> BookingState:
    """Handle confirmation stage."""
    state["stage"] = "confirm"
    slot = state["selected_slot"]
    date = state["selected_date"] or "Today"

    message = f"""Review your appointment:

**Date:** {date}
**Time:** {slot}

Confirm or Cancel?"""

    options = ["Confirm", "Cancel", "Change Slot"]

    # Ensure the confirmation message is in history before interrupting
    if not state["messages"] or state["messages"][-1].get("content") != message:
        state["messages"].append({
            "role": "assistant",
            "content": message,
            "options": options
        })

    user_choice = interrupt({
        "role": "assistant",
        "content": message,
        "available_options": options
    })

    state["messages"].append({
        "role": "user",
        "content": user_choice
    })

    return state

def collect_details_node(state: BookingState) -> BookingState:
    """Collect patient details before final booking."""

    msg_name = "Please enter your full name:"
    if state["messages"][-1]["content"] != msg_name:
        state["messages"].append({"role": "assistant", "content": msg_name})
    name = interrupt(msg_name)
    state["messages"].append({"role": "user", "content": name})

    msg_phone = "Please enter your phone number:"
    if state["messages"][-1]["content"] != msg_phone:
        state["messages"].append({"role": "assistant", "content": msg_phone})
    phone = interrupt(msg_phone)
    state["messages"].append({"role": "user", "content": phone})

    state["customer_name"] = name
    state["customer_phone"] = phone
    state["stage"] = "completed"

    return state

def completed_node(state: BookingState) -> BookingState:
    """Finalize booking and insert into database."""
    state["stage"] = "completed"
    slot = state["selected_slot"]
    date = state["selected_date"]

    message = f"""✅ Appointment Confirmed!

                **Date:** {date}
                **Time:** {slot}

                Thank you."""

    state["messages"].append({
        "role": "assistant",
        "content": message
    })

    state["available_options"] = []
    state["stage"] = "completed"

    return state

def cancelled_node(state: BookingState) -> BookingState:
    """Handle cancelled booking."""
    state["messages"].append({
        "role": "assistant",
        "content": "Thank you for connecting. Send 'hi' to restart your booking.",
        "options": ["Book Again"]
    })

    # state["available_options"] = ["Book Again"]
    state["stage"] = "cancelled"

    return state


def build_booking_graph():
    workflow = StateGraph(BookingState)

    # 1. Add Nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("select_date", select_date_node)
    workflow.add_node("select_slot", select_slot_node)
    workflow.add_node("confirm", confirm_node)
    workflow.add_node("collect_details", collect_details_node)
    workflow.add_node("completed", completed_node)
    workflow.add_node("cancelled", cancelled_node)

    # 2. Set Entry Point
    workflow.set_entry_point("greeting")

    # 3. Add Conditional Edges WITH MAPPING
    # Format: add_conditional_edges(source_node, routing_function, mapping_dict)

    # The greeting node routes based on user intent
    workflow.add_edge("greeting", "select_date")
    workflow.add_edge("select_date", "select_slot")
    workflow.add_edge("select_slot", "confirm")

    workflow.add_conditional_edges(
        "confirm",
        llm_router,
        {
            "confirm": "confirm",
            "collect_details": "collect_details",
            "cancelled": "cancelled",
            "select_slot": "select_slot"
        }
    )

    workflow.add_edge("collect_details", "completed")

    # 4. Final Edges to END
    workflow.add_edge("completed", END)
    workflow.add_edge("cancelled", END)

    # 5. Compile with Checkpointer
    return workflow.compile(checkpointer=MemorySaver())


# Create the compiled graph
booking_graph = build_booking_graph()


def process_message(state: BookingState, user_message: str, thread_id: str = "default_session") -> BookingState:
    """Process a user message through the booking graph."""
    config = {"configurable": {"thread_id": thread_id}}

    # Check if the graph is currently interrupted
    current_state = booking_graph.get_state(config)

    if current_state.tasks and current_state.tasks[0].interrupts:
        # Resume the graph with the user's message
        result = booking_graph.invoke(Command(resume=user_message), config=config)
    else:
        # No interrupt, so start/continue normally
        # Add user message to state (unless it's an initial trigger)
        if user_message.lower() != "hi" or state["messages"]:
            # Avoid duplicate user messages if already added
            if not state["messages"] or state["messages"][-1].get("content") != user_message:
                state["messages"].append({
                    "role": "user",
                    "content": user_message
                })
        # Run the graph
        result = booking_graph.invoke(state, config=config)

    return result