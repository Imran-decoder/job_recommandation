import json
import os
from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# --- Configuration ---
load_dotenv()

app = FastAPI(title="Career Recommendation Engine")

# --- Database Setup ---
mongo_uri = os.getenv("MONGO_URI")
neo4j_uri = os.getenv("URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("AUTH")

# --- Load Mapping ---
try:
    with open("mapping.json", "r") as f:
        NSQF_MAPPING = json.load(f)
except FileNotFoundError:
    NSQF_MAPPING = {"skills": {}, "interestSectors": {}, "qualifications": {}, "streams": {}, "careerAspirations": {}}

# --- Connect Drivers ---
try:
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    cli = MongoClient(mongo_uri)
    db = cli["career-advisor"]
    users_collection = db["userdetails"]
    print("✅ Databases Connected")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

# --- Load Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI Model loaded")


class UserRequest(BaseModel):
    user_id: str


# --- Logic Functions ---

def resolve_user_nsqf_sectors(user_doc):
    matched_sectors = set()
    # (Mapping logic remains same as before, shortened here for brevity)
    for skill in user_doc.get("skills", []):
        matched_sectors.update(NSQF_MAPPING["skills"].get(skill, []))
    for sec in user_doc.get("interestSectors", []):
        matched_sectors.update(NSQF_MAPPING["interestSectors"].get(sec, []))
    highest_q = user_doc.get("education", {}).get("highestQualification", "")
    if highest_q: matched_sectors.update(NSQF_MAPPING["qualifications"].get(highest_q, []))
    career_goal = user_doc.get("careerGoal", "")
    if career_goal: matched_sectors.update(NSQF_MAPPING["careerAspirations"].get(career_goal, []))
    return list(matched_sectors)


def find_similar_qualifications_with_rank(user_embedding, target_sectors):
    query = """
    MATCH (q:Qualification) 
    WHERE q.sector_name IN $sectors AND q.embedding IS NOT NULL
    WITH q, vector.similarity.cosine(q.embedding, $userVector) AS score
    ORDER BY score DESC
    LIMIT 10
    RETURN q.title AS title, 
           q.sector_name AS sector,
           q.nsqf_level AS nsqf_level, 
           score
    """
    with neo4j_driver.session() as session:
        result = session.run(query, userVector=user_embedding, sectors=target_sectors)

        data = []
        # Enumerate gives us 0, 1, 2... we add 1 to make it Rank 1, 2, 3...
        for rank, record in enumerate(result, start=1):
            data.append({
                "rank": rank,  # <--- NEW: Added Ranking
                "title": record["title"],
                "sector": record.get("sector", "Unknown"),
                "level": record.get("nsqf_level", "Unknown"),
                "score": record["score"]
            })
        return data


def save_graph_relationships(user_id_str, recommendations):
    if not recommendations: return 0

    query = """
    MERGE (u:User {mongoId: $userId})
    WITH u
    UNWIND $batch AS item
    MATCH (q:Qualification {title: item.title})
    MERGE (u)-[r:RECOMMENDED_FOR]->(q)
    SET r.score = item.score,
        r.rank = item.rank,         // <--- NEW: Saving Rank to Graph
        r.generated_at = datetime()
    RETURN count(r) as relationships_created
    """
    try:
        with neo4j_driver.session() as session:
            result = session.run(query, userId=str(user_id_str), batch=recommendations)
            return result.single()["relationships_created"]
    except Exception as e:
        print(f"Graph Error: {e}")
        return 0


# --- API Endpoint (Streaming) ---

@app.post("/api/recommend/stream")
async def recommend_stream(payload: UserRequest):
    async def process_pipeline():
        try:
            # Step 1: Log
            yield json.dumps({"type": "log", "message": "🔍 Fetching User Profile from MongoDB..."}) + "\n"

            if not ObjectId.is_valid(payload.user_id):
                yield json.dumps({"type": "error", "message": "Invalid ID format"}) + "\n"
                return

            user_oid = ObjectId(payload.user_id)
            user_doc = users_collection.find_one({"$or": [{"_id": user_oid}, {"user": user_oid}]})

            if not user_doc:
                yield json.dumps({"type": "error", "message": "User not found"}) + "\n"
                return

            # Step 2: Log
            yield json.dumps({"type": "log", "message": "⚙️ Resolving Industry Sectors..."}) + "\n"
            matched_sectors = resolve_user_nsqf_sectors(user_doc)
            if not matched_sectors:
                matched_sectors = user_doc.get("interestSectors", [])  # Fallback

            # Step 3: Log
            yield json.dumps({"type": "log", "message": "🧠 Generating AI Embeddings..."}) + "\n"
            career_goal = user_doc.get("careerGoal", "")
            skills = ", ".join(user_doc.get("skills", []))
            quiz_responses = user_doc.get("quizResponses", [])
            learning_context = " ".join([f"{q.get('question', '')} {q.get('answer', '')}" for q in quiz_responses])
            text_to_embed = f"Career Goal: {career_goal}. Skills: {skills}. Context: {learning_context}"

            user_embedding = model.encode(text_to_embed).tolist()

            # Step 4: Log
            yield json.dumps({"type": "log", "message": "🕸️ Querying Graph for Top 10 Recommendations..."}) + "\n"
            recs_data = find_similar_qualifications_with_rank(user_embedding, matched_sectors)

            # Step 5: Log
            yield json.dumps({"type": "log", "message": f"💾 Saving {len(recs_data)} relationships to Neo4j..."}) + "\n"
            save_count = save_graph_relationships(payload.user_id, recs_data)

            # Step 6: Final Result
            final_response = {
                "status": "success",
                "user_id": payload.user_id,
                "resolved_sectors": matched_sectors,
                "saved_relationships": save_count,
                "recommendations": recs_data
            }
            yield json.dumps({"type": "result", "data": final_response}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    # Return as Streaming Response (NDJSON format)
    return StreamingResponse(process_pipeline(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)