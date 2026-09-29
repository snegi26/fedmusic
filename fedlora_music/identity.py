"""Create this client's SuperNode identity key pair (ECDSA P-384, OpenSSH format).

The private key never leaves the client. Send the ``.pub`` file to the federation
operator, who registers it with ``flwr supernode register <file> <connection>``.

    python -m fedlora_music.identity --out-dir <data-dir>/keys
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

KEY_NAME = "supernode_key"


def ensure_identity(out_dir: Path) -> tuple[Path, str]:
    """Return ``(private_key_path, public_key_text)``, creating the pair if missing."""
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    priv, pub = out_dir / KEY_NAME, out_dir / f"{KEY_NAME}.pub"
    if not priv.is_file():
        key = ec.generate_private_key(ec.SECP384R1())
        fd = os.open(priv, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.OpenSSH,
                    serialization.NoEncryption(),
                )
            )
        pub.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
            )
            + b"\n"
        )
    return priv, pub.read_text().strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    _, public = ensure_identity(args.out_dir.expanduser().resolve())
    print(public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
