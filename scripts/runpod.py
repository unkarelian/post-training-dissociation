"""Minimal RunPod GPU helper.
Reads RUNPOD_API_KEY from environment or ~/.env.
Usage:
  python3 scripts/runpod.py up ["NVIDIA A40"] [SECURE|COMMUNITY]
  python3 scripts/runpod.py status
  python3 scripts/runpod.py ssh
  python3 scripts/runpod.py down
"""
import json, os, pathlib, sys, urllib.request


def get_key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"].strip()
    env_path = os.path.expanduser("~/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Set RUNPOD_API_KEY in environment or ~/.env")


KEY = get_key()


def q(query):
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        json.dumps({"query": query}).encode(),
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "curl/8"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


def pods():
    return q(
        "query { myself { clientBalance pods { id name desiredStatus costPerHr "
        "runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } "
        "machine { gpuDisplayName } } } }"
    )["data"]["myself"]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "up":
        gpu = sys.argv[2] if len(sys.argv) > 2 else "NVIDIA A40"
        cloud = sys.argv[3] if len(sys.argv) > 3 else "SECURE"
        pub_path = os.environ.get("SSH_PUB_KEY", os.path.expanduser("~/.ssh/id_ed25519.pub"))
        if not os.path.exists(pub_path):
            pub_path = os.path.expanduser("~/.ssh/runpod_claude.pub")
        pub = pathlib.Path(pub_path).read_text().strip()
        m = (
            'mutation { podFindAndDeployOnDemand(input: {cloudType: %s, gpuCount: 1, volumeInGb: 0, '
            'containerDiskInGb: 70, gpuTypeId: "%s", name: "face-of-another", '
            'imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", '
            'ports: "22/tcp", env: [{key: "PUBLIC_KEY", value: "%s"}]}) '
            '{ id costPerHr machine { gpuDisplayName } } }' % (cloud, gpu, pub)
        )
        print(json.dumps(q(m), indent=1))
    elif cmd == "status":
        print(json.dumps(pods(), indent=1))
    elif cmd == "ssh":
        for p in pods()["pods"]:
            for pt in (p.get("runtime") or {}).get("ports") or []:
                if pt["privatePort"] == 22 and pt["isIpPublic"]:
                    print(f"{pt['ip']} {pt['publicPort']}")
    elif cmd == "down":
        for p in pods()["pods"]:
            print(q('mutation { podTerminate(input: {podId: "%s"}) }' % p["id"]))
        print("Remaining balance:", pods()["clientBalance"])
