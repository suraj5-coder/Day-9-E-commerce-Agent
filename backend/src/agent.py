# Day 9 – E-commerce Agent (ACP Style)
# Store: The Agentic Store
# Features: Catalog Browsing, Cart Management, Order Persistence (JSON)

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("acp_commerce_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# 1. Product Catalog (The Agentic Store)
# -------------------------
CATALOG = [
    {
        "id": "hoodie-dev-blk",
        "name": "Developer Hoodie (Black)",
        "description": "Premium cotton hoodie with 'Ship It' printed on the back.",
        "price": 1499,
        "currency": "INR",
        "category": "apparel",
        "color": "black",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "tee-acp-wht",
        "name": "ACP Protocol Tee",
        "description": "White t-shirt featuring the Agentic Commerce logo.",
        "price": 799,
        "currency": "INR",
        "category": "apparel",
        "color": "white",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "mug-neural",
        "name": "Neural Network Mug",
        "description": "Ceramic mug that reveals code when hot liquid is poured.",
        "price": 499,
        "currency": "INR",
        "category": "accessories",
        "color": "black",
    },
    {
        "id": "cap-tech",
        "name": "Tech Stack Cap",
        "description": "Minimalist cap for developers.",
        "price": 599,
        "currency": "INR",
        "category": "accessories",
        "color": "navy",
        "sizes": ["One Size"],
    },
    {
        "id": "sticker-pack",
        "name": "Laptop Sticker Pack",
        "description": "Pack of 10 dev-themed stickers.",
        "price": 199,
        "currency": "INR",
        "category": "accessories",
        "color": "multi",
    },
]

# -------------------------
# 2. Persistence (JSON)
# -------------------------
ORDERS_FILE = "orders.json"

# Ensure orders file exists
if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w") as f:
        json.dump([], f)

def _load_all_orders() -> List[Dict]:
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def _save_order(order: Dict):
    orders = _load_all_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

# -------------------------
# 3. User Session State
# -------------------------
@dataclass
class Userdata:
    cart: List[Dict] = field(default_factory=list)  # list of {product_id, quantity, attrs}

# -------------------------
# 4. Helper Logic
# -------------------------
def list_products(query: str = None, category: str = None) -> List[Dict]:
    results = []
    for p in CATALOG:
        match = True
        if category and p['category'] != category:
            match = False
        if query:
            q = query.lower()
            if q not in p['name'].lower() and q not in p['description'].lower():
                match = False
        if match:
            results.append(p)
    return results

def find_product_fuzzy(ref_text: str) -> Optional[Dict]:
    """Simple logic to find a product based on user speech."""
    ref = ref_text.lower()
    
    # 1. Try exact ID
    for p in CATALOG:
        if p['id'] == ref: return p

    # 2. Try Name contains
    for p in CATALOG:
        if p['name'].lower() in ref: return p
    
    # 3. Try keywords (e.g. "the hoodie", "the mug")
    for p in CATALOG:
        keywords = p['name'].lower().split()
        if any(k in ref for k in keywords if len(k) > 3):
            return p
            
    return None

def calculate_total(cart):
    total = 0
    for item in cart:
        p = next((x for x in CATALOG if x["id"] == item["product_id"]), None)
        if p:
            total += p["price"] * item["quantity"]
    return total

# -------------------------
# 5. Agent Tools
# -------------------------

@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    query: Annotated[Optional[str], Field(description="Search term (e.g. 'hoodie', 'mug')")] = None,
    category: Annotated[Optional[str], Field(description="Category (apparel, accessories)")] = None,
) -> str:
    """Browse the store catalog."""
    products = list_products(query, category)
    if not products:
        return "I couldn't find any items matching that description."
    
    lines = [f"Found {len(products)} items in The Agentic Store:"]
    for p in products[:5]: # Limit to 5 for voice clarity
        lines.append(f"- {p['name']} ({p['price']} INR)")
    
    return "\n".join(lines) + "\n\nWhich one would you like to add to your cart?"

@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: Annotated[str, Field(description="The name or reference of the product to add")],
    quantity: Annotated[int, Field(description="Quantity")] = 1,
    size: Annotated[Optional[str], Field(description="Size (if applicable)")] = None,
) -> str:
    """Add an item to the shopping cart."""
    product = find_product_fuzzy(product_ref)
    if not product:
        return f"I'm not sure which product you meant by '{product_ref}'. Could you be more specific?"
    
    ctx.userdata.cart.append({
        "product_id": product["id"],
        "name": product["name"],
        "quantity": quantity,
        "size": size
    })
    
    return f"Added {quantity} x {product['name']} to your cart. Cart Total: {calculate_total(ctx.userdata.cart)} INR."

@function_tool
async def view_cart(ctx: RunContext[Userdata]) -> str:
    """Check what is currently in the cart."""
    if not ctx.userdata.cart:
        return "Your cart is currently empty."
    
    lines = ["Your Cart:"]
    for item in ctx.userdata.cart:
        details = f"Size: {item['size']}" if item['size'] else ""
        lines.append(f"- {item['quantity']} x {item['name']} {details}")
    
    lines.append(f"Total: {calculate_total(ctx.userdata.cart)} INR")
    return "\n".join(lines)

@function_tool
async def place_order(ctx: RunContext[Userdata]) -> str:
    """Confirm and place the order, saving it to the backend system."""
    if not ctx.userdata.cart:
        return "You cannot place an empty order."
    
    order_id = f"ORD-{str(uuid.uuid4())[:6].upper()}"
    total = calculate_total(ctx.userdata.cart)
    
    order_data = {
        "order_id": order_id,
        "timestamp": datetime.utcnow().isoformat(),
        "items": ctx.userdata.cart,
        "total_amount": total,
        "currency": "INR",
        "status": "CONFIRMED"
    }
    
    _save_order(order_data)
    
    # Clear cart
    ctx.userdata.cart = []
    
    return f"Order placed successfully! Your Order ID is {order_id}. Total amount: {total} INR. Is there anything else I can help you with?"

@function_tool
async def get_last_order(ctx: RunContext[Userdata]) -> str:
    """Fetch the most recent order details."""
    orders = _load_all_orders()
    if not orders:
        return "You haven't placed any orders yet."
    
    last = orders[-1]
    return f"Your last order ({last['order_id']}) contained {len(last['items'])} items for a total of {last['total_amount']} INR."

# -------------------------
# 6. Agent Configuration
# -------------------------
class CommerceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are 'Alex', the intelligent sales assistant for 'The Agentic Store'.
            We sell developer swag like Hoodies, Tees, and Tech Accessories.
            
            Your Role:
            1. Help users browse the catalog using `show_catalog`.
            2. Add items to their cart using `add_to_cart`.
            3. Answer questions about the cart using `view_cart`.
            4. Finalize the purchase using `place_order`.
            5. If asked about history, use `get_last_order`.

            Style: Professional, efficient, and polite. Keep responses concise for voice.
            """,
            tools=[show_catalog, add_to_cart, view_cart, place_order, get_last_order],
        )

# -------------------------
# 7. Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try: proc.userdata["vad"] = silero.VAD.load()
    except: pass

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("🚀 STARTING AGENTIC COMMERCE STORE")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    await session.start(
        agent=CommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))