#!/usr/bin/env python3
"""
ESPN FastCast WebSocket Proof of Concept
========================================
Connects to ESPN's real-time push notification system (FastCast)
to receive live score updates via WebSocket instead of polling.

NOT FOR PRODUCTION - Experimental/Research only.

Protocol reverse-engineered from ESPN's FITT JS bundle:
  Discovery: https://fastcast.semfs.engsvc.go.com/public/websockethost
  WebSocket: wss://{host}:{port}/FastcastService/pubsub/profiles/{profileId}
  Opcodes: B=heartbeat, C=connect, H=checkpoint, M=maintenance,
           P=publish, R=replay, S=subscribe, U=unsubscribe
  Subscribe: {"op":"S", "sid":"<session_id>", "tc":"<topic>"}
  
  College baseball sportId = 14
"""

import asyncio
import json
import sys
import time
import ssl
import urllib.request
import zlib

DISCOVERY_URL = "https://fastcast.semfs.engsvc.go.com/public/websockethost"
PROFILE_ID = 12000

# Known topic patterns from ESPN's FITT framework
# The header scoreboard uses sportTopics with slugs
TOPICS_TO_TRY = [
    "gp-baseball-college-baseball-scoreboard",
    "sport-14",
    "college-baseball",
    "gp-college-baseball",
    "gp-baseball-14",
    "baseball-college-baseball",
]


def discover_ws_host():
    """Get WebSocket host + token from ESPN's discovery endpoint."""
    print(f"[*] Discovering FastCast host...")
    req = urllib.request.Request(DISCOVERY_URL)
    req.add_header("User-Agent", "Mozilla/5.0")
    
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    
    print(f"[+] Host: {data['ip']}")
    print(f"[+] Secure port: {data['securePort']}")
    return data


def try_decompress(payload):
    """Try to decompress zlib/gzip payload from FastCast."""
    if isinstance(payload, str):
        try:
            import base64
            decoded = base64.b64decode(payload)
            return zlib.decompress(decoded).decode('utf-8')
        except:
            pass
    return payload


async def connect_fastcast(host_info, duration=180):
    """Connect to FastCast WebSocket and subscribe to college baseball topics."""
    try:
        import websockets
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
        import websockets

    token = host_info["token"]
    host = host_info["ip"]
    port = host_info["securePort"]
    
    ws_url = f"wss://{host}:{port}/FastcastService/pubsub/profiles/{PROFILE_ID}?TrafficManager-Token={token}"
    print(f"\n[*] Connecting to FastCast WebSocket...")
    
    ssl_ctx = ssl.create_default_context()
    session_id = None
    
    try:
        async with websockets.connect(ws_url, ssl=ssl_ctx, ping_interval=None) as ws:
            print("[+] Connected!")
            
            # Send CONNECT
            await ws.send(json.dumps({"op": "C"}))
            
            start = time.time()
            msg_count = 0
            subscribed_topics = set()
            
            while time.time() - start < duration:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=15)
                    msg_count += 1
                    elapsed = time.time() - start
                    
                    try:
                        data = json.loads(msg)
                        op = data.get("op", "?")
                        
                        # Handle CONNECT response
                        if op == "C":
                            session_id = data.get("sid")
                            rc = data.get("rc")
                            hbi = data.get("hbi", 30)
                            print(f"  [{elapsed:.1f}s] CONNECT: rc={rc}, sid={session_id}, hbi={hbi}s")
                            
                            if rc == 200 and session_id:
                                # Subscribe to all topic patterns
                                for topic in TOPICS_TO_TRY:
                                    sub_msg = {"op": "S", "sid": session_id, "tc": topic}
                                    await ws.send(json.dumps(sub_msg))
                                    print(f"  [{elapsed:.1f}s] -> Subscribed to: {topic}")
                                    subscribed_topics.add(topic)
                            continue
                        
                        # Handle HEARTBEAT
                        if op == "B":
                            await ws.send(json.dumps({"op": "B", "sid": session_id}))
                            continue
                        
                        # Handle SUBSCRIBE response
                        if op == "S":
                            tc = data.get("tc", "?")
                            rc = data.get("rc", "?")
                            print(f"  [{elapsed:.1f}s] SUBSCRIBE response: topic={tc}, rc={rc}")
                            if rc != 200:
                                print(f"    [!] Subscribe FAILED for {tc}")
                            continue
                        
                        # Handle UNSUBSCRIBE
                        if op == "U":
                            tc = data.get("tc", "?")
                            print(f"  [{elapsed:.1f}s] UNSUBSCRIBED: {tc}")
                            continue
                        
                        # Handle CHECKPOINT (initial data snapshot)
                        if op == "H":
                            tc = data.get("tc", "?")
                            mid = data.get("mid", "?")
                            pl = data.get("pl")
                            print(f"\n  [{elapsed:.1f}s] ===== CHECKPOINT for {tc} (mid={mid}) =====")
                            if pl:
                                payload = try_decompress(pl) if isinstance(pl, str) else pl
                                if isinstance(payload, str):
                                    try:
                                        payload = json.loads(payload)
                                    except:
                                        pass
                                if isinstance(payload, dict):
                                    print(f"    Keys: {list(payload.keys())[:15]}")
                                    # Look for score-related data
                                    for key in ['events', 'sports', 'scores', 'games', 'competitions']:
                                        if key in payload:
                                            val = payload[key]
                                            if isinstance(val, list):
                                                print(f"    {key}: {len(val)} items")
                                                if val:
                                                    first = val[0]
                                                    if isinstance(first, dict):
                                                        print(f"      First item keys: {list(first.keys())[:10]}")
                                            elif isinstance(val, dict):
                                                print(f"    {key} keys: {list(val.keys())[:10]}")
                                    # Print a preview
                                    pl_str = json.dumps(payload, indent=2)
                                    if len(pl_str) > 2000:
                                        print(f"    Preview (first 2000 chars):\n{pl_str[:2000]}...")
                                    else:
                                        print(f"    Full payload:\n{pl_str}")
                                elif isinstance(payload, str):
                                    # Could be a CDN URL for checkpoint data
                                    if payload.startswith("http"):
                                        print(f"    CDN Checkpoint URL: {payload}")
                                    else:
                                        print(f"    String payload ({len(payload)} chars): {payload[:500]}")
                                else:
                                    print(f"    Payload type: {type(payload).__name__}")
                            continue
                        
                        # Handle PUBLISH (live update!)
                        if op == "P":
                            tc = data.get("tc", "?")
                            mid = data.get("mid", "?")
                            pl = data.get("pl")
                            print(f"\n  [{elapsed:.1f}s] ★★★ PUBLISH on {tc} (mid={mid}) ★★★")
                            if pl:
                                payload = try_decompress(pl) if isinstance(pl, str) else pl
                                if isinstance(payload, str):
                                    try:
                                        payload = json.loads(payload)
                                    except:
                                        pass
                                if isinstance(payload, (dict, list)):
                                    pl_str = json.dumps(payload, indent=2)
                                    print(f"    {pl_str[:1000]}")
                                else:
                                    print(f"    {str(payload)[:500]}")
                            continue
                        
                        # Handle MAINTENANCE
                        if op == "M":
                            print(f"  [{elapsed:.1f}s] MAINTENANCE: {data}")
                            continue
                        
                        # Unknown opcode
                        print(f"  [{elapsed:.1f}s] Unknown op={op}: {json.dumps(data)[:300]}")
                        
                    except json.JSONDecodeError:
                        print(f"  [{elapsed:.1f}s] Non-JSON ({len(msg)} bytes): {msg[:200]}")
                
                except asyncio.TimeoutError:
                    elapsed = time.time() - start
                    print(f"  [{elapsed:.1f}s] ... waiting (active topics: {len(subscribed_topics)}) ...")
            
            print(f"\n[*] Finished. {msg_count} messages in {duration}s")
    
    except Exception as e:
        print(f"[!] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("=" * 60)
    print("ESPN FastCast WebSocket - Proof of Concept v2")
    print("=" * 60)
    
    host_info = discover_ws_host()
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print(f"\n[*] Will listen for {duration}s, trying {len(TOPICS_TO_TRY)} topic patterns...")
    
    await connect_fastcast(host_info, duration=duration)


if __name__ == "__main__":
    asyncio.run(main())
