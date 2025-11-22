from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
E2B_API_KEY = os.getenv("E2B_API_KEY")
from e2b_code_interpreter import Sandbox,SandboxQuery, SandboxState, SandboxInfo
import json

# def sandbox_to_dict(sb):
#     return {
#         "sandbox_id": sb.sandbox_id,
#         "template_id": sb.template_id,
#         "name": sb.name,
#         "metadata": sb.metadata,
#         "started_at": sb.started_at.isoformat() if isinstance(sb.started_at, datetime) else sb.started_at,
#         "end_at": sb.end_at.isoformat() if isinstance(sb.end_at, datetime) else sb.end_at,
#         "state": sb.state.value if hasattr(sb.state, "value") else sb.state,
#         "cpu_count": sb.cpu_count,
#         "memory_mb": sb.memory_mb,
#         "envd_version": sb.envd_version,
#     }

# def extract_all_sandbox_metadata(paginator):
#     sandboxes = paginator.next_items()
#     return [sandbox_to_dict(sb) for sb in sandboxes]

# def print_paginator_details(pg):
#     print("\n--- Paginator Metadata ---")
#     for attr in dir(pg):
#         if attr.startswith("_"):
#             continue
#         try:
#             print(f"{attr}: {getattr(pg, attr)}")
#         except:
#             print(f"{attr}: <unreadable>")

#     print("\n--- Sandbox Items Metadata ---")
#     items = pg.next_items()
#     for i, item in enumerate(items):
#         print(f"\nSandbox #{i+1}")
#         try:
#             print(item.__dict__)
#         except:
#             import inspect
#             print(inspect.getmembers(item))


from e2b import Sandbox

def get_all_sandboxes():
    p = Sandbox.list()
    items = []
    while p.has_next:
        items.extend(p.next_items())
    return items

running_sandboxes = get_all_sandboxes()

if not running_sandboxes:
    print("No running sandboxes found — skipping kill operation.")
else:
    for sb in running_sandboxes:
        print(f"Killing sandbox {sb.id}")
        sb.kill()

