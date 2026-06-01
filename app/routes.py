import json 
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from .task import create_task, get_task, cancel_task, TaskStatus


logger = logging.getLogger(__name__)
router = APIRouter()


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    max_research_loops: int = Field(default=2, ge=1, le=5)


@router.post("/research")
async def create_research(req: ResearchRequest, request: Request):
    """Kick off a research task. Returns task_id immediately; the graph runs in the background."""
    deps = request.app.state.deps
    task_id = await create_task(
        redis=deps.redis,
        query=req.query,
        max_research_loops=req.max_research_loops,
    )
    return {"task_id": task_id, "status": TaskStatus.PENDING}


@router.get("/research/{task_id}")
async def research_status(task_id: str, request: Request):
    """Get current status"""
    deps = request.app.state.deps
    record = await get_task(deps.redis, task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return record


@router.get("/research/{task_id}/events")
async def research_polling(task_id: str, request: Request, since: int = 0):
    """Polling endpoint. Returns events from `since` onward + current task state + report"""
    deps = request.app.state.deps
    record = await get_task(deps.redis, task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")

    raw = await deps.redis.lrange(f"events:{task_id}", since, -1)
    events = [json.loads(e) for e in raw]

    return {
        "events": events,
        "next_cursor": since + len(events),
        "status": record["status"],
        "report": record.get("final_report") if record["status"] == "complete" else None,
        "error": record.get("error", "") if record["status"] == 
        "failed" else None,
    }

# just for debugging/
@router.get("/research/{task_id}/artifacts")
async def research_artifacts(task_id: str, request: Request):
    redis = request.app.state.deps.redis
    raw = await redis.get(f"dpr:artifacts:{task_id}")
    if raw is None:
        raise HTTPException(404, "artifacts not found (task may not exist or has expired)")
    return Response(content=raw, media_type="application/json")

@router.delete("/research/{task_id}")
async def research_cancel(task_id: str, request: Request):
    """Cancel a running task."""
    deps = request.app.state.deps
    cancelled = await cancel_task(deps.redis, task_id)
    if not cancelled:
        record = await get_task(deps.redis, task_id)
        if record is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {"task_id": task_id, "status": record["status"], "message": "task was not running"}
    return {"task_id": task_id, "status": "cancelling"}