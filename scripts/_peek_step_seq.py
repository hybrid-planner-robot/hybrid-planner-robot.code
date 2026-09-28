#!/usr/bin/env python3
"""
_peek_step_seq.py — Runs INSIDE the Docker container.

Read the latched /vlm_planner/step_complete message (if any) and print its
seq so the host can sync before waiting on a new inject.

Stdout JSON: {"seq": <int>}  (seq=-1 if none within timeout)
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


class _Peek(Node):
    def __init__(self) -> None:
        super().__init__("_peek_step_seq")
        self.seq = -1
        from rclpy.qos import DurabilityPolicy, QoSProfile

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/vlm_planner/step_complete", self._on, qos)

    def _on(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.seq = int(data.get("seq", -1))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=1.5)
    args = parser.parse_args()

    rclpy.init()
    node = _Peek()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t0 = time.time()
    while node.seq < 0 and (time.time() - t0) < args.timeout:
        executor.spin_once(timeout_sec=0.05)
        # Even if we already got a latched msg, spin a bit for discovery.
        if node.seq >= 0 and (time.time() - t0) > 0.2:
            break
    # One more spin in case latched arrived at t≈0
    for _ in range(5):
        executor.spin_once(timeout_sec=0.05)
    seq = node.seq
    node.destroy_node()
    rclpy.shutdown()
    print(json.dumps({"seq": seq}))


if __name__ == "__main__":
    main()
