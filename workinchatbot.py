from dotenv import load_dotenv
import os
import json
import numpy as np

from google.oauth2 import service_account
from llama_index.core import SQLDatabase
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from sqlalchemy import create_engine

from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
env_path = r"C:\Users\HP\Videos\AI Projects\Chatbot LLama\.env"
load_dotenv(dotenv_path=env_path)

# --- Load credentials ---
credentials_path = os.getenv("bq_service_accout_key")
project_id = os.getenv("bq_project_id")
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# --- Set up SQLAlchemy engine for BigQuery ---
engine = create_engine(
    f"bigquery://{project_id}",
    credentials_path=credentials_path
)

# --- Table metadata ---
tables = {
    "tbproddb.Teachers_Data": (
        "You need to generate query for BigQuery",
        "For counting total teachers, school, sector we need to use count(distinct user_id)",
        "This table contains information about teachers data such as their name, Gender (Female Or Male), signed_in_status ('Signed In', 'Not Signed In'), sector, coach assigned and EMIS. The six sectors are Tarnol, Nilore, B.K (aka Barakahu), Sihala, Urban-I, and Urban-II.", 
    )
}

# --- Load LLM & Embeddings ---
llm = Groq(
    model="llama3-70b-8192",
    system_prompt="You are an SQL expert that can  understand data. Whever giving the count use distinct user_id to count the users by Sector. If asked for graph you can give data in the following format only without explanation so that a graph can be generated  " \
    """{
  "type": "bar",
  "data": {
    "labels": ["January", "February", "March", "April", "May", "June", "July", "August"],
    "datasets": [
      {
        "label": "Cosmic Sales",
        "data": [120, 150, 180, 170, 200, 220, 210, 230],
        "backgroundColor": [
          "#7F00FF",
          "#E100FF",
          "#FF6EC7",
          "#00FFFF",
          "#FFD700",
          "#FF4500",
          "#32CD32",
          "#1E90FF"
        ],
        "borderRadius": 10,
        "barPercentage": 0.6,
        "hoverBackgroundColor": "#FFFFFF",
        "hoverBorderWidth": 2
      }
    ]
  },
  "options": {
    "responsive": true,
    "plugins": {
      "title": {
        "display": true,
        "text": "🌌 Interstellar Revenue Overview 🌠",
        "color": "#ffffff",
        "font": {
          "size": 24,
          "weight": "bold",
          "family": "Arial"
        },
        "padding": {
          "top": 10,
          "bottom": 20
        }
      },
      "tooltip": {
        "backgroundColor": "#222222",
        "titleColor": "#ffffff",
        "bodyColor": "#cccccc",
        "borderColor": "#ffffff",
        "borderWidth": 1
      },
      "legend": {
        "display": false
      },
      "datalabels": {
        "display": true,
        "anchor": "end",
        "align": "end",
        "color": "#fff",
        "backgroundColor": "rgba(0,0,0,0.7)",
        "borderRadius": 4,
        "font": {
          "weight": "bold",
          "size": 12
        },
        "padding": 6
      }
    },
    "scales": {
      "x": {
        "ticks": {
          "color": "#ffffff",
          "font": {
            "weight": "bold"
          }
        },
        "grid": {
          "display": false
        }
      },
      "y": {
        "beginAtZero": true,
        "ticks": {
          "color": "#ffffff"
        },
        "grid": {
          "color": "#444444"
        }
      }
    },
    "animation": {
      "duration": 2000,
      "easing": "easeOutBounce"
    }
  },
  "plugins": ["chartjs-plugin-datalabels"]
}"""
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# --- Set up query engine ---
sql_database = SQLDatabase(engine, include_tables=tables, sample_rows_in_table_info=50)
query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=tables, llm=llm)

# --- Chat Memory Handling ---
MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

def print_chat_history(memory):
    print("\n--- Chat History ---")
    for chat in memory:
        print(f"User: {chat['user']}\nBot: {chat['bot']}\n")

# --- Semantic Similarity Search ---
def find_similar_query(current_query, memory, threshold=0.85):
    if not memory:
        return None

    query_embedding = embed_model.get_text_embedding(current_query)
    similarities = []

    for entry in memory:
        past_embedding = embed_model.get_text_embedding(entry["user"])
        sim_score = cosine_similarity([query_embedding], [past_embedding])[0][0]
        similarities.append(sim_score)

    max_sim = max(similarities)
    if max_sim >= threshold:
        best_match = memory[similarities.index(max_sim)]
        return best_match
    return None

# --- Correction Detection ---
def is_correction(feedback):
    correction_keywords = ["wrong", "no", "actually", "not correct", "misunderstood", "should be", "that was incorrect"]
    return any(keyword in feedback.lower() for keyword in correction_keywords)

def revise_memory_entry(last_user_query, old_response, user_feedback):
    prompt = f"""You are correcting your past response.
    
Earlier, the user asked: "{last_user_query}"
You answered: "{old_response}"
The user now says: "{user_feedback}"

Based on this correction, please provide an updated response in place of the old one. Keep it clear and accurate."""
    
    revised = llm.complete(prompt)
    return revised.text.strip()

# --- Main Query Function with Correction Handling ---
def ask_query(query, query_engine, memory):
    print("\nThinking...\n")

    if memory:
        last_entry = memory[-1]
    else:
        last_entry = None

    if is_correction(query) and last_entry:
        print("🛠️ Correction detected. Fixing memory...")

        revised_response = revise_memory_entry(
            last_user_query=last_entry["user"],
            old_response=last_entry["bot"],
            user_feedback=query
        )

        print("✅ Revised response:", revised_response)

        # Update last memory entry
        last_entry["bot"] = revised_response
        save_memory(memory)
        return

    # Proceed with normal flow
    similar = find_similar_query(query, memory)

    if similar:
        print("🔁 Found similar past query. Reusing context.")
        context = f"Earlier the user asked: '{similar['user']}' and the assistant replied: '{similar['bot']}'.\nNow answer this: {query}"
    else:
        context = query

    try:
        response = query_engine.query(context)
        memory.append({"user": query, "bot": response.response})
        save_memory(memory)
        print("SQL Query:", response.metadata["sql_query"])
        print("Response:", response.response)
    except Exception as e:
        print("An error occurred:", str(e))

# --- Main Loop ---
if __name__ == "__main__":
    print("🧠 Welcome to the Self-Healing SQL Chatbot (BigQuery + LLaMA3)")
    memory = load_memory()
    print_chat_history(memory)

    while True:
        user_input = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        ask_query(user_input, query_engine, memory)