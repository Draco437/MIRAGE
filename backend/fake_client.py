"""
fake_client.py - pretends to be a VR headset (or any sensor client) so you can test the
WHOLE live pipeline with no Unity, no headset and no webcam:

    log in -> create a session -> open the live connections -> stream fake gaze,
    hand and answer data -> watch the ML pipeline react.

Run from backend/ with the server AND the Celery worker running:
    pip install websockets          (once)
    python fake_client.py --scenario distracted
Scenarios: engaged, distracted, stressed.
"""
import argparse
import asyncio
import getpass
import json
import random
import sys
import time
import urllib.error
import urllib.request


def api(base, path, body=None, token=None):
    """Tiny REST helper (standard library only)."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, headers=headers, method="POST" if body is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        sys.exit(f"{path} failed: HTTP {exc.code} {exc.read().decode()[:300]}")
    except urllib.error.URLError as exc:
        sys.exit(f"Cannot reach {base} ({exc.reason}). Is the daphne server running?")


def pick_target(scenario, rng):
    r = rng.random()
    if scenario == "engaged":
        return "eyes" if r < 0.45 else "face" if r < 0.85 else "body"
    if scenario == "distracted":
        return "away" if r < 0.7 else "object" if r < 0.95 else "body"
    return rng.choice(["eyes", "away", "object", "body", "face", "away"])  # stressed: jumpy


MOTOR = {  # (hand speed, jitter) typical for each scenario
    "engaged": (0.30, 0.05),
    "distracted": (0.20, 0.08),
    "stressed": (0.60, 0.25),
}
ACCURACY = {"engaged": 0.85, "distracted": 0.5, "stressed": 0.4}


async def send_loop(ws, scenario, seconds, rng):
    start = time.monotonic()
    next_motor = 0.0
    next_answer = 4.0
    count = 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= seconds:
            break
        t = round(elapsed, 2)
        await ws.send(json.dumps({"type": "gaze_data", "t": t, "target": pick_target(scenario, rng)}))
        if elapsed >= next_motor:
            speed, jitter = MOTOR[scenario]
            await ws.send(json.dumps({
                "type": "motor_data", "t": t,
                "hand_speed": max(0.0, rng.gauss(speed, speed * 0.2)),
                "jitter": max(0.0, rng.gauss(jitter, jitter * 0.2)),
            }))
            next_motor = elapsed + 0.5
        if elapsed >= next_answer:
            await ws.send(json.dumps({
                "type": "response_event", "t": t, "prompt_id": "fake_prompt",
                "response_time_ms": rng.uniform(1500, 5000),
                "correct": rng.random() < ACCURACY[scenario],
            }))
            next_answer = elapsed + 8.0
        count += 1
        await asyncio.sleep(0.1)
    return count


def show(name, message):
    kind = message.get("type")
    if name == "unity":
        if kind == "welcome":
            print(f"[unity]     welcome, starting settings: {message['settings']}")
        elif kind == "adaptation":
            who = message.get("source")
            print(f"[unity]     SCENE CHANGE ({who}, {message.get('label')}): {message.get('reason')}")
            print(f"            new settings: {message['settings']}")
        else:
            print(f"[unity]     {message}")
    else:
        if kind == "live" and message.get("kind") == "engagement":
            print(f"[dashboard] engagement: {message['label']} "
                  f"(confidence {message['confidence']:.2f}, model: {message['source']})")
        elif kind == "live" and message.get("kind") == "session_status":
            print(f"[dashboard] session is now {message['status']}")
        elif kind == "live" and message.get("kind") == "adaptation":
            pass  # already shown on the unity side
        else:
            print(f"[dashboard] {message}")


async def listen(ws, name):
    try:
        async for raw in ws:
            show(name, json.loads(raw))
    except Exception:
        pass  # connection closed


async def run(args, unity_url, dashboard_url, access):
    import websockets

    async with websockets.connect(dashboard_url + "?token=" + access) as dashboard:
        async with websockets.connect(unity_url) as unity:
            listeners = [
                asyncio.create_task(listen(dashboard, "dashboard")),
                asyncio.create_task(listen(unity, "unity")),
            ]
            print(f"Streaming '{args.scenario}' data for {args.seconds} seconds...\n")
            sent = await send_loop(unity, args.scenario, args.seconds, random.Random(7))
            await asyncio.sleep(3)  # let the last results arrive
            for task in listeners:
                task.cancel()
    print(f"\nDone. Sent about {sent} gaze samples. The session is now completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["engaged", "distracted", "stressed"], default="distracted")
    parser.add_argument("--seconds", type=int, default=40)
    parser.add_argument("--username", default="therapist")
    parser.add_argument("--password", default=None)
    parser.add_argument("--base", default="http://localhost:8000")
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"Password for {args.username}: ")
    login = api(args.base, "/api/auth/login/", {"username": args.username, "password": password})
    access = login["access"]

    children = api(args.base, "/api/children/", token=access)
    if children:
        child = children[0]
    else:
        child = api(args.base, "/api/children/", {"name": "Test child", "age": 7}, access)
    session = api(args.base, "/api/sessions/", {"child_id": child["id"], "scenario": "greeting"}, access)
    print(f"Session created: {session['session_id']}")

    try:
        asyncio.run(run(args, session["unity_ws_url"], session["dashboard_ws_url"], access))
    except ImportError:
        sys.exit("The 'websockets' package is missing. Run: pip install websockets")
    except Exception as exc:
        sys.exit(f"Live connection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()