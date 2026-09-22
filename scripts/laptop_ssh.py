"""
Helper tool to securely execute remote commands on Hasin's laptop (DESKTOP-R6QPSH5)
over Tailscale via SSH, adhering to all workspace rules and verification checks.
"""

import sys
import paramiko

LAPTOP_IP = "100.120.206.30"
LAPTOP_USER = "hasin"
LAPTOP_PASS = "070409"
EXPECTED_HOSTNAME = "DESKTOP-R6QPSH5"


def run_laptop_cmd(cmd: str, timeout: int = 60):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Try key first if configured, fallback to password
        client.connect(
            LAPTOP_IP,
            username=LAPTOP_USER,
            password=LAPTOP_PASS,
            timeout=10,
            look_for_keys=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to {LAPTOP_IP}: {e}", file=sys.stderr)
        return 1

    # Mandatory step: Always verify hostname first
    stdin, stdout, stderr = client.exec_command("hostname")
    remote_host = stdout.read().decode().strip()
    _ = stdout.channel.recv_exit_status()

    if remote_host.upper() != EXPECTED_HOSTNAME.upper():
        print(
            f"[FATAL] Hostname mismatch! Expected {EXPECTED_HOSTNAME}, got {remote_host}. Aborting!",
            file=sys.stderr
        )
        client.close()
        return 1

    print(f"[VERIFIED HOST: {remote_host}] Executing: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    
    out = stdout.read().decode()
    err = stderr.read().decode()
    exit_status = stdout.channel.recv_exit_status()

    if out:
        print(out, end="")
    if err:
        print(err, file=sys.stderr, end="")

    client.close()
    return exit_status


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/laptop_ssh.py <command>")
        # Default test command
        cmd = "hostname"
    else:
        cmd = " ".join(sys.argv[1:])
    
    sys.exit(run_laptop_cmd(cmd))
